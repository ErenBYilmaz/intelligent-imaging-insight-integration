import unittest

from pynetdicom import StoragePresentationContexts


class TestPresentationContexts(unittest.TestCase):
    def test_printing_context_names(self):
        contexts = StoragePresentationContexts
        for context in contexts:
            print(context.abstract_syntax.name)