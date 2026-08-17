import unittest

from document_profile import compute_profile


class ComputeProfileTests(unittest.TestCase):
    def test_short_document(self):
        profile = compute_profile("A short note.", pages=1)
        self.assertEqual(profile.characters, 13)
        self.assertEqual(profile.estimated_tokens, 3)
        self.assertEqual(profile.pages, 1)
        self.assertEqual(profile.median_chars_per_page, 13.0)
        self.assertEqual(profile.doc_type, "unknown")

    def test_markdown_headings(self):
        text = "# Introduction\nText\n## Method\nMore\n### Results\nDone"
        profile = compute_profile(text)
        self.assertEqual(profile.sections, 3)

    def test_table_heavy_document(self):
        text = "Name | Value | Unit\n--- | --- | ---\nA | 1 | kg\nConclusion"
        profile = compute_profile(text)
        self.assertTrue(profile.table_heavy)

    def test_empty_and_near_empty_documents(self):
        empty = compute_profile("", pages=2)
        self.assertEqual(empty.characters, 0)
        self.assertEqual(empty.median_chars_per_page, 0.0)
        self.assertFalse(empty.table_heavy)
        self.assertFalse(empty.list_heavy)

        near_empty = compute_profile(" ")
        self.assertEqual(near_empty.estimated_tokens, 0)
        self.assertIsNone(near_empty.median_chars_per_page)


if __name__ == "__main__":
    unittest.main()
