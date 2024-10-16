import warnings
import re
from dotenv import load_dotenv
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import SummaryIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.tools import FunctionTool
from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.llms import LLM
from typing import Optional
from pydantic import BaseModel
from llama_index.core.schema import Document
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.prompts import PromptTemplate
import logging
import json
from llama_index.core.workflow import draw_all_possible_flows

warnings.filterwarnings('ignore')
_ = load_dotenv()

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
)

data_dir = "data"
data_out_dir = "data_out_rfp"

files = ["azure_gov.pdf", "azure_wiki.pdf", "msft_10k_2024.pdf", "msft_ddr.pdf"]

file_dicts = {}

for f in files:
    file_base = Path(f).stem
    full_file_path = str(Path(data_dir) / f)

    file_docs = parser.load_data(full_file_path)

    # attach metadata
    for idx, d in enumerate(file_docs):
        d.metadata["file_path"] = f
        d.metadata["page_num"] = idx + 1

    file_dicts[f] = {"file_path": full_file_path, "docs": file_docs}

summary_llm = OpenAI(model="gpt-4o-mini")

for f in files:
    print(f">> Generate summary for file {f}")
    index = SummaryIndex(file_dicts[f]["docs"])
    response = index.as_query_engine(llm=summary_llm).query(
        "Generate a short 1-2 line summary of this file to help inform an agent on what this file is about."
    )
    print(f">> Generated summary: {str(response)}")
    file_dicts[f]["summary"] = str(response)

persist_dir = "storage_rfp_chroma"

vector_store = ChromaVectorStore.from_params(
    collection_name="rfp_docs", persist_dir=persist_dir
)

index = VectorStoreIndex.from_vector_store(vector_store)

# run this only if the Chroma index is not already built

# all_nodes = [c for d in file_dicts.values() for c in d["docs"]]
# index.insert_nodes(all_nodes)

# function tools
def generate_tool(file: str, file_description: Optional[str] = None):
    """Return a function that retrieves only within a given file."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="file_path", operator=FilterOperator.EQ, value=file),
        ]
    )

    def chunk_retriever_fn(query: str) -> str:
        retriever = index.as_retriever(similarity_top_k=5, filters=filters)
        nodes = retriever.retrieve(query)

        full_text = "\n\n========================\n\n".join(
            [n.get_content(metadata_mode="all") for n in nodes]
        )

        return full_text

    # define name as a function of the file
    fn_name = Path(file).stem + "_retrieve"

    tool_description = f"Retrieves a small set of relevant document chunks from {file}."
    if file_description is not None:
        tool_description += f"\n\nFile Description: {file_description}"

    tool = FunctionTool.from_defaults(
        fn=chunk_retriever_fn, name=fn_name, description=tool_description
    )

    return tool

# generate tools
tools = []
for f in files:
    tools.append(generate_tool(f, file_description=file_dicts[f]["summary"]))

print(tools[0].metadata)

# Build Workflow
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


# this is the research agent's system prompt, tasked with answering a specific question
AGENT_SYSTEM_PROMPT = """\
You are a research agent tasked with filling out a specific form key/question with the appropriate value, given a bank of context.
You are given a specific form key/question. Think step-by-step and use the existing set of tools to help answer the question.

You MUST always use at least one tool to answer each question. Only after you've determined that existing tools do not \
answer the question should you try to reason from first principles and prior knowledge to answer the question.

You MUST try to answer the question instead of only saying 'I dont know'.

"""

# This is the prompt tasked with extracting information from an RFP file.
EXTRACT_KEYS_PROMPT = """\
You are provided an entire RFP document, or a large subsection from it.

We wish to generate a response to the RFP in a way that adheres to the instructions within the RFP, \
including the specific sections that an RFP response should contain, and the content that would need to go \
into each section.

Your task is to extract out a list of "questions", where each question corresponds to a specific section that is required
in the RFP response.
Put another way, after we extract out the questions we will go through each question and answer each one \
with our downstream research assistant, and the combined
question:answer pairs will constitute the full RFP response.

- Make sure the questions are comprehensive and adheres to the RFP requirements.
- Make sure each question is descriptive - this gives our downstream assistant context to fill out the value for that
question
- Extract out all the questions as a list of strings.

"""

# this is the prompt that generates the final RFP response given the original template text and question-answer pairs.
GENERATE_OUTPUT_PROMPT = """\
You are an expert analyst.
Your task is to generate an RFP response according to the given RFP and question/answer pairs.

You are given the following RFP and qa pairs:

<rfp_document>
{output_template}
</rfp_document>

<question_answer_pairs>
{answers}
</question_answer_pairs>

Not every question has an appropriate answer. This is because the agent tasked with answering the question did not have
the right context to answer it.
If this is the case, you MUST come up with an answer that is reasonable. You CANNOT say that you are unsure in any area
of the RFP response.


Please generate the output according to the template and the answers, in markdown format.
Directly output the generated markdown content, do not add any additional text, such as "```markdown" or "Here is the
output:".
Follow the original format of the template as closely as possible, and fill in the answers into the appropriate sections.
"""

class OutputQuestions(BaseModel):
    """List of keys that make up the sections of the RFP response."""
    questions: List[str]

class OutputTemplateEvent(Event):
    docs: List[Document]

class QuestionsExtractedEvent(Event):
    questions: List[str]

class HandleQuestionEvent(Event):
    question: str

class QuestionAnsweredEvent(Event):
    question: str
    answer: str

class CollectedAnswersEvent(Event):
    combined_answers: str

class LogEvent(Event):
    msg: str
    delta: bool = False

class RFPWorkflow(Workflow):
    """RFP workflow."""

    def __init__(
        self,
        tools,
        parser: LlamaParse,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = data_out_dir,
        agent_system_prompt: str = AGENT_SYSTEM_PROMPT,
        generate_output_prompt: str = GENERATE_OUTPUT_PROMPT,
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(**kwargs)
        self.tools = tools

        self.parser = parser

        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.similarity_top_k = similarity_top_k

        self.output_dir = output_dir

        self.agent_system_prompt = agent_system_prompt

        # if not exists, create
        out_path = Path(self.output_dir) / "workflow_output"
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)

        self.generate_output_prompt = PromptTemplate(generate_output_prompt)

    @step
    async def parse_output_template(self, ctx: Context, ev: StartEvent) -> OutputTemplateEvent:
        # Change the file extension to .json
        out_template_path = Path(f"{self.output_dir}/workflow_output/output_template.json")

        if out_template_path.exists():
            with open(out_template_path, "r") as f:
                docs = [Document(**doc) for doc in json.load(f)]
        else:
            docs = await self.parser.aload_data(ev.rfp_template_path)
            # save output template to file
            with open(out_template_path, "w") as f:
                json.dump([doc.dict() for doc in docs], f)

        await ctx.set("output_template", docs)
        return OutputTemplateEvent(docs=docs)

    @step
    async def extract_questions(
            self, ctx: Context, ev: OutputTemplateEvent
    ) -> HandleQuestionEvent:
        docs = ev.docs

        # save all_questions to file
        out_keys_path = Path(f"{self.output_dir}/workflow_output/all_keys.txt")
        if out_keys_path.exists():
            with open(out_keys_path, "r") as f:
                output_qs = f.read().splitlines()
        else:
            # try stuffing all text into the prompt
            all_text = "\n\n".join([d.get_content(metadata_mode="all") for d in docs])
            prompt = PromptTemplate(template=EXTRACT_KEYS_PROMPT)

            try:
                # Collect the response from the async generator
                response = ""
                async for chunk in self.llm.astream(prompt, all_text=all_text):
                    response += chunk

                # Debugging: Print the response
                print("LLM Response:")
                print(response)

                # Parse the response into questions
                # Assuming each question is on a new line
                output_qs = response.strip().split('\n')

                # If the LLM uses numbering or bullets, adjust parsing accordingly
                # For example, remove numbering
                output_qs = [re.sub(r'^\d+\.\s*', '', q).strip() for q in output_qs]

            except Exception as e:
                print(f"Error during LLM response collection: {e}")
                output_qs = []

            with open(out_keys_path, "w") as f:
                f.write("\n".join(output_qs))

        await ctx.set("num_to_collect", len(output_qs))

        for question in output_qs:
            ctx.send_event(HandleQuestionEvent(question=question))

        return None

    @step
    async def handle_question(
            self, ctx: Context, ev: HandleQuestionEvent
    ) -> QuestionAnsweredEvent:
        question = ev.question

        # initialize a Function Calling "research" agent where given a task, it can pull responses from relevant tools and synthesize over it
        research_agent = FunctionCallingAgentWorker.from_tools(
            tools, llm=llm, system_prompt=self.agent_system_prompt
        ).as_agent()
        research_agent.verbose = False

        # ensure the agent's memory is cleared
        response = await research_agent.aquery(question)

        if self._verbose:
            # instead of printing the message directly, write the event to stream!
            msg = f">> Asked question: {question}\n>> Got response: {str(response)}"
            ctx.write_event_to_stream(LogEvent(msg=msg))

        return QuestionAnsweredEvent(question=question, answer=str(response))

    @step
    async def combine_answers(
            self, ctx: Context, ev: QuestionAnsweredEvent
    ) -> CollectedAnswersEvent:
        num_to_collect = await ctx.get("num_to_collect")
        results = ctx.collect_events(ev, [QuestionAnsweredEvent] * num_to_collect)
        if results is None:
            return None

        combined_answers = "\n".join([result.model_dump_json() for result in results])
        # save combined_answers to file
        with open(
                f"{self.output_dir}/workflow_output/combined_answers.jsonl", "w"
        ) as f:
            f.write(combined_answers)

        return CollectedAnswersEvent(combined_answers=combined_answers)

    @step
    async def generate_output(
            self, ctx: Context, ev: CollectedAnswersEvent
    ) -> StopEvent:
        output_template = await ctx.get("output_template")
        output_template = "\n".join(
            [doc.get_content("none") for doc in output_template]
        )

        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> GENERATING FINAL OUTPUT"))

        resp = await self.llm.astream(
            self.generate_output_prompt,
            output_template=output_template,
            answers=ev.combined_answers,
        )

        final_output = ""
        async for r in resp:
            ctx.write_event_to_stream(LogEvent(msg=r, delta=True))
            final_output += r

        # save final_output to file
        with open(f"{self.output_dir}/workflow_output/final_output.md", "w") as f:
            f.write(final_output)

        return StopEvent(result=final_output)


llm = OpenAI(model="gpt-4o-mini")

workflow = RFPWorkflow(
    tools,
    parser=parser,
    llm=llm,
    verbose=True,
    timeout=None,  # don't worry about timeout to make sure it completes
)

draw_all_possible_flows(RFPWorkflow, filename="rfp_workflow.html")

import asyncio

async def main():

    handler = workflow.run(rfp_template_path=str(Path(data_dir) / "jedi_cloud_rfp.pdf"))

    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            if event.delta:
                print(event.msg, end="")
            else:
                print(event.msg)

    response = await handler
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())
