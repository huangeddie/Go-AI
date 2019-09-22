import unittest
from go_ai import data

greedy_policy_args = {
    'mode': 'greedy'
}
random_policy_args = {
    'mode': 'random'
}


class MyTestCase(unittest.TestCase):
    def test_rand_vs_greed(self):
        win_rate = data.make_episodes(5, random_policy_args, greedy_policy_args,
                                      32, num_workers=4)
        self.assertAlmostEqual(win_rate, 0, delta=0.1)

    def test_greed_vs_rand(self):
        win_rate = data.make_episodes(5, greedy_policy_args, random_policy_args,
                                      32, num_workers=4)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)


if __name__ == '__main__':
    unittest.main()
