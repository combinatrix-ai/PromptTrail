import logging
import unittest

from prompttrail.agent.session_transformers import (
    MetadataTransformer,
    SessionTransformer,
)
from prompttrail.core import Metadata, Session

logger = logging.getLogger(__name__)


class TestSessionTransformer(unittest.TestCase):
    def test_transform(self):
        session = Session()
        hook = SessionTransformer()
        with self.assertRaises(NotImplementedError):
            hook.process(session)


class TestMetadataTransformer(unittest.TestCase):
    def test_transform(self):
        session = Session()
        transform_hook = MetadataTransformer()
        with self.assertRaises(NotImplementedError):
            transform_hook.process_metadata(session.metadata, session)

        class TestMetadataTransformer(MetadataTransformer):
            def process_metadata(self, metadata, session) -> Metadata:
                metadata["test_key"] = "test_value"
                return metadata

        transform_hook = TestMetadataTransformer()
        result_metadata = transform_hook.process_metadata(session.metadata, session)
        self.assertEqual(result_metadata["test_key"], "test_value")
        result = transform_hook.process(session)
        self.assertEqual(result.metadata["test_key"], "test_value")


if __name__ == "__main__":
    unittest.main()
