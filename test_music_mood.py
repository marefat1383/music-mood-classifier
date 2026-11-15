import unittest

from music_mood import predict_mood


class MusicMoodTests(unittest.TestCase):
    def test_calm_song(self):
        self.assertEqual(predict_mood(80, 0.2), "calm")

    def test_invalid_energy_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_mood(120, 1.2)


if __name__ == "__main__":
    unittest.main()
