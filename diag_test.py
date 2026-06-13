import asyncio
from uuid import uuid4
import sys

from src.shared.database import AsyncSessionLocal
from src.delivery.infrastructure.models import UserModel
from src.ml_engine.infrastructure.models import ProfileModel

from src.genai.infrastructure.langchain_chain import get_llm_service
from src.ml_engine.application.use_cases import ProfileUserFromCVUseCase
from src.ml_engine.infrastructure.cluster_repository import SQLClusterRepository
from src.ml_engine.infrastructure.cv_parser import LocalCVParserService
from src.ml_engine.infrastructure.embeddings import get_embedding_service
from src.ml_engine.infrastructure.user_profile_repository import SQLUserProfileRepository

class MockCVParser(LocalCVParserService):
    async def extract_text(self, content: bytes, content_type: str) -> str:
        return "This is a valid software engineering CV. I know python and docker."

async def run():
    print("Connecting to DB...")
    async with AsyncSessionLocal() as session:
        print("Instantiating use case...")
        from src.ml_engine.infrastructure.skill_repository import SQLSkillRepository
        use_case = ProfileUserFromCVUseCase(
            cv_parser=MockCVParser(),
            embedding_service=get_embedding_service(),
            cluster_repository=SQLClusterRepository(session),
            profile_repository=SQLUserProfileRepository(session),
            llm_service=get_llm_service(),
            skill_repository=SQLSkillRepository(session)
        )
        try:
            print("Executing use case...")
            # We need to insert a mock user first
            test_user_id = uuid4()
            user = UserModel(user_id=test_user_id, email=f"test_{test_user_id}@test.com", full_name="Test User")
            session.add(user)
            await session.flush()
            
            await use_case.execute(
                user_id=test_user_id,
                cv_id=uuid4(),
                cv_content=b"dummy content",
                content_type="application/pdf"
            )
            print("Success!")
        except Exception as e:
            print(f"Exception during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await session.rollback()

if __name__ == "__main__":
    # Ensure Windows console uses utf-8
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(run())
