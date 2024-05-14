from vidur.config import Config
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main():
    config = Config()

    set_seeds(config.seed)

    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    main()
