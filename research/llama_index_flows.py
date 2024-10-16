import warnings
from dotenv import load_dotenv
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

warnings.filterwarnings('ignore')
_ = load_dotenv()

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")

async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())