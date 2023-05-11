import pong
import submitted
import sys
import numpy as np


def main():
    state_quantization = [10, 10, 2, 2, 10]
    if len(sys.argv) > 1 and sys.argv[1] == "traditional":
        learner = submitted.q_learner(0.05, 0.05, 0.99, 5, state_quantization)
        states = [
            [x, y, vx, vy, py]
            for x in range(10)
            for y in range(10)
            for vx in range(2)
            for vy in range(2)
            for py in range(10)
        ]
    else:
        learner = submitted.deep_q(1e-3, 0.05, 0.99, 5)
        states = []

    pong_game = pong.PongGame(
        learner=learner, visible=False, state_quantization=state_quantization
    )
    pong_game.run(m_games=np.inf, states=states)
    learner.save("trained_model.pkl")


if __name__ == "__main__":
    main()
