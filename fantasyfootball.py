import time
from random import gauss
import multiprocessing as mp
from copy import deepcopy


class Player(object):
    def __init__(self, name, wins, losses, ties, pf, pa, mu, sigma):
        self.name = name
        self.wins = wins
        self.losses = losses
        self.ties = ties
        self.pf = pf
        self.pa = pa
        self.mu = mu
        self.sigma = sigma

    def play_against(self, opp):
        my_score = round(gauss(self.mu, self.sigma), 2)
        opp_score = round(gauss(self.mu, self.sigma), 2)
        if my_score > opp_score:
            self.wins += 1
            opp.losses += 1
        elif my_score < opp_score:
            self.losses += 1
            opp.wins += 1
        else:
            self.ties += 1
            opp.ties += 1
        self.pf += my_score
        self.pa += opp_score
        opp.pa += my_score
        opp.pf += opp_score

    def games_played(self):
        return self.wins + self.losses + self.ties

    def comparator(self):
        return [self.wins, -1 * self.losses, -1 * self.ties, self.pf, -1 * self.pa, self.name]

    def __str__(self):
        return "{}: {:d}-{:d}-{:d} | PF: {:.2f} | PA: {:.2f}".format(self.name, self.wins, self.losses, self.ties, self.pf, self.pa)


def get_current_standings():
    with open("ff.csv", "r") as f:
        raw = [line.strip().split(",") for line in f]
        raw = raw[1:]

        orig = {}
        for i in raw:
            i[2:5] = map(int, i[2:5])
            i[5:] = map(float, i[5:])
            orig[i[1]] = Player(i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8])
    return orig


def get_schedule():
    ans = []
    with open("ff_schedule.csv", "r") as f:
        for line in f:
            matchup = line.strip().split(",")
            ans.append(tuple(matchup[1:]))
    return ans


def simulate_season(orig, schedule):
    plays = deepcopy(orig)

    for matchup in schedule:
        p1 = matchup[0]
        p2 = matchup[1]
        plays[p1].play_against(plays[p2])

    comps = sorted([plays[p].comparator() for p in plays], reverse=True)
    comps = comps[:4] + comps[8:]

    return comps


def run_mc(orig, schedule, rnds, output):
    playoffs = {}
    toilets = {}
    for p in orig:
        playoffs[p] = 0
        toilets[p] = 0

    for _ in range(rnds):
        res = simulate_season(orig, schedule)
        for i, c in enumerate(res):
            if i < 4:
                playoffs[c[-1]] += 1
            else:
                toilets[c[-1]] += 1

    output.put((playoffs, toilets))


def print_results(p, t, r):
    print("MONTE CARLO SIMULATION ({:,} ROUNDS):\n".format(r))
    print("Likelihood of making the playoffs:")
    for val, name in p:
        print("{}: {:.3f}%".format(name, val / r * 100))

    print()
    print("Likelihood of making the toilet bowl:")
    for val, name in t:
        print("{}: {:.3f}%".format(name, val / r * 100))


def find_probs(rnds, workers):
    if rnds % workers:
        rnds += workers + (rnds % workers)

    orig = get_current_standings()
    schedule = get_schedule()

    output = mp.Queue()
    processes = [mp.Process(target=run_mc, args=(orig, schedule, rnds // workers, output,)) for _ in range(workers)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    playoffs_overall = dict(zip(orig.keys(), [0] * len(orig)))
    toilet_overall = dict(zip(orig.keys(), [0] * len(orig)))

    for _ in range(workers):
        results = output.get()
        playoffs = results[0]
        toilets = results[1]
        for player in orig:
            playoffs_overall[player] += playoffs[player]
            toilet_overall[player] += toilets[player]

    playoffs_overall = reversed(sorted([(playoffs_overall[p], p) for p in playoffs_overall]))
    toilet_overall = reversed(sorted([(toilet_overall[p], p) for p in toilet_overall]))

    print_results(playoffs_overall, toilet_overall, rnds)


def main():
    start = time.time()

    rnds = 10**6
    workers = 8

    find_probs(rnds, workers)

    elapsed = time.time() - start

    if elapsed < 1:
        elapsed *= 1000
        text = "milliseconds"
    else:
        text = "seconds"
    print("Program took:", elapsed, text)


if __name__ == '__main__':
    main()
