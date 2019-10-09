import unittest

import go_ai.distributed
from go_ai import data

greedy_policy_args = {
    'mode': 'greedy',
    'board_size': 5
}
random_policy_args = {
    'mode': 'random',
    'board_size': 5
}


class MyTestCase(unittest.TestCase):
    def test_rand_vs_greed(self):
        win_rate = go_ai.distributed.make_episodes(random_policy_args, greedy_policy_args,
                                                   32, num_workers=4)
        self.assertAlmostEqual(win_rate, 0, delta=0.1)

    def test_greed_vs_rand(self):
        win_rate = go_ai.distributed.make_episodes(greedy_policy_args, random_policy_args,
                                                   32, num_workers=4)
        self.assertAlmostEqual(win_rate, 1, delta=0.1)


if __name__ == '__main__':
    unittest.main()
