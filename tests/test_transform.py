import unittest
from src.transform import Profile, normalize_results, extract_profile_data, parse_name


class TestTransform(unittest.TestCase):
    def test_profile_model(self):
        """Test the Profile pydantic model."""
        # Create a profile with minimal data
        profile = Profile(linkedin_url="https://www.linkedin.com/in/johndoe/")
        self.assertEqual(profile.linkedin_url, "https://www.linkedin.com/in/johndoe/")
        self.assertIsNone(profile.first_name)
        self.assertIsNone(profile.last_name)
        
        # Create a profile with full data
        profile = Profile(
            linkedin_url="https://www.linkedin.com/in/johndoe/",
            title="Software Engineer",
            first_name="John",
            last_name="Doe",
            description="Experienced developer",
            profile_image_url="https://example.com/image.jpg"
        )
        self.assertEqual(profile.title, "Software Engineer")
        self.assertEqual(profile.first_name, "John")
        self.assertEqual(profile.last_name, "Doe")
        self.assertEqual(profile.description, "Experienced developer")
        self.assertEqual(profile.profile_image_url, "https://example.com/image.jpg")
    
    def test_normalize_results_empty(self):
        """Test normalizing an empty result set."""
        profiles = normalize_results([])
        self.assertEqual(len(profiles), 0)
    
    def test_normalize_results_deduplication(self):
        """Test that duplicate LinkedIn URLs are deduplicated."""
        raw_items = [
            {"link": "https://www.linkedin.com/in/johndoe/", "title": "John Doe | LinkedIn"},
            {"link": "https://www.linkedin.com/in/johndoe/", "title": "John Doe - Software Engineer"},
            {"link": "https://www.linkedin.com/in/janedoe/", "title": "Jane Doe | LinkedIn"}
        ]
        profiles = normalize_results(raw_items)
        self.assertEqual(len(profiles), 2)  # Only two unique URLs
        
        # Check that URLs are as expected
        urls = [p.linkedin_url for p in profiles]
        self.assertIn("https://www.linkedin.com/in/johndoe/", urls)
        self.assertIn("https://www.linkedin.com/in/janedoe/", urls)
    
    def test_parse_name_from_metatags(self):
        """Test name parsing when metatags are available."""
        result = parse_name("John", "Doe", "Whatever Title")
        self.assertEqual(result["first_name"], "John")
        self.assertEqual(result["last_name"], "Doe")
    
    def test_parse_name_from_title(self):
        """Test name parsing from title when metatags are not available."""
        result = parse_name("", "", "John Doe | LinkedIn")
        self.assertEqual(result["first_name"], "John")
        self.assertEqual(result["last_name"], "Doe")
        
        result = parse_name("", "", "John Doe - Software Engineer")
        self.assertEqual(result["first_name"], "John")
        self.assertEqual(result["last_name"], "Doe")
    
    def test_extract_profile_data(self):
        """Test profile data extraction from a search result."""
        # Test with minimal data
        item = {
            "link": "https://www.linkedin.com/in/johndoe/",
            "title": "John Doe | LinkedIn",
            "snippet": "Software Engineer with 5+ years experience."
        }
        metatags = {}
        profile = extract_profile_data(item, metatags, "https://www.linkedin.com/in/johndoe/")
        
        self.assertEqual(profile.linkedin_url, "https://www.linkedin.com/in/johndoe/")
        self.assertEqual(profile.title, "John Doe | LinkedIn")
        self.assertEqual(profile.first_name, "John")
        self.assertEqual(profile.last_name, "Doe")
        self.assertEqual(profile.description, "Software Engineer with 5+ years experience.")
        
        # Test with full metatags
        metatags = {
            "profile:first_name": "John",
            "profile:last_name": "Doe",
            "profile:title": "Senior Software Engineer",
            "og:description": "Experienced software engineer with a passion for Python",
            "twitter:image": "https://example.com/image.jpg"
        }
        profile = extract_profile_data(item, metatags, "https://www.linkedin.com/in/johndoe/")
        
        self.assertEqual(profile.title, "Senior Software Engineer")
        self.assertEqual(profile.first_name, "John")
        self.assertEqual(profile.last_name, "Doe")
        self.assertEqual(profile.description, "Experienced software engineer with a passion for Python")
        self.assertEqual(profile.profile_image_url, "https://example.com/image.jpg")


if __name__ == "__main__":
    unittest.main() 