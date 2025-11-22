import random
import matplotlib.pyplot as plt

class Menace:
    def __init__(self): 
        self.matchboxes = {}

    def get_beads(self, state):
        if state not in self.matchboxes:
            valid_moves = [i for i, x in enumerate(state) if x == 0]
            self.matchboxes[state] = {move: 3 for move in valid_moves}
        return self.matchboxes[state]

    def choose_move(self, state):
        beads = self.get_beads(state)
        total_beads = sum(beads.values())
        
        if total_beads == 0:
            valid_moves = [i for i, x in enumerate(state) if x == 0]
            beads = {move: 3 for move in valid_moves}
            self.matchboxes[state] = beads
            total_beads = sum(beads.values())

        r = random.randint(1, total_beads)
        cumulative = 0
        for move, count in beads.items():
            cumulative += count
            if r <= cumulative:
                return move

    def play_game(self, opponent):
        game_history = []
        state = (0,) * 9
        
        for turn in range(9):
            if turn % 2 == 0:
                move = self.choose_move(state)
                game_history.append((state, move))
                player = 1
            else:
                move = opponent.choose_move(state)
                player = 2
            
            new_state = list(state)
            new_state[move] = player
            state = tuple(new_state)
            
            result = self.check_winner(state)
            if result:
                self.reinforce(game_history, result)
                return result
        
        self.reinforce(game_history, "draw")
        return "draw"

    def check_winner(self, state):
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in wins:
            if state[a] == state[b] == state[c] and state[a] != 0:
                return "win" if state[a] == 1 else "loss"
        return None

    def reinforce(self, history, result):
        if result == "win": delta = 3
        elif result == "loss": delta = -1
        else: delta = 1

        for state, move in history:
            beads = self.matchboxes[state]
            beads[move] = max(0, beads[move] + delta)

class RandomOpponent:
    def choose_move(self, state):
        return random.choice([i for i, x in enumerate(state) if x == 0])

menace = Menace()
opponent = RandomOpponent()

total_games = 5000
batch_size = 500

x_epochs = []
y_win_rates = []
wins_in_batch = 0

print(f"{'Games Played':<15} | {'Wins':<10} | {'Win Rate (%)':<15}")
print("-" * 45)

for i in range(1, total_games + 1):
    res = menace.play_game(opponent)
    if res == "win":
        wins_in_batch += 1
    if i % batch_size == 0:
        win_rate = wins_in_batch / batch_size
        percentage = win_rate * 100
        y_win_rates.append(win_rate)
        x_epochs.append(i)
        print(f"{i:<15} | {wins_in_batch:<10} | {percentage:.2f}%")
        wins_in_batch = 0

plt.figure(figsize=(10, 6))
plt.plot(x_epochs, y_win_rates, color='green', label='Win Rate', linewidth=2)

plt.title(f'MENACE Win Rate over {total_games} Games')
plt.xlabel('Number of Games Played')
plt.ylabel('Win Rate (0.0 - 1.0)')
plt.ylim(0, 1.0) 
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()