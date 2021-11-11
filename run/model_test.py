import unittest

import model

class TestModel(unittest.TestCase):

    def test_get_random_seeds(self) -> None:
        seeds = model.getRandomSeeds(random_seed=0, num_seeds=5)

        self.assertIn(0, seeds)
        self.assertEqual(len(seeds), 5)

    def test_get_model_names(self) -> None:
        names = model.getModelNames(mode='monomer', use_ptm=False, num_models=3)
        self.assertEqual(names, ('model_1', 'model_2', 'model_3'))

        names = model.getModelNames(mode='monomer', use_ptm=True, num_models=2)
        self.assertEqual(names, ('model_1_ptm', 'model_2_ptm'))

        names = model.getModelNames(mode='multimer', num_models=5)
        self.assertEqual(names, ('model_1_multimer', 'model_2_multimer',
                                 'model_3_multimer', 'model_4_multimer',
                                 'model_5_multimer'))

    def test_get_model_runner(self) -> None:
        # Just testing to make sure that it can run.
        model_runner = model.getModelRunner(model_name='model_1_ptm')


if __name__ == '__main__':
    unittest.main()
