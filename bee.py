import numpy as np
import random
import copy


class Bee:
    def __init__(self):
        self.solution = None
        self.function_value = 0
        self.fitness = 0
        self.trial = 0
        self.probability = 0
        self.weight = 0

    def discard_items_over_capacity(self, data, capacity, item_ranking):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)
        for index, probability in item_ranking:
            if current_weight > capacity:
                if self.solution[int(index)] == 1:
                    self.solution[int(index)] = 0
                    current_weight -= weights[int(index)]
            else:
                self.weight = current_weight
                break

    def discard_items_over_capacity2(self, data, capacity):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)
        while(current_weight>capacity):
            idx = random.choice(np.argwhere(self.solution == 1).reshape(-1))
            self.solution[idx] = 0
            current_weight -= weights[idx]
        self.weight = current_weight

    def set_random_solution(self, data, capacity, item_ranking):
        self.solution = np.array([random.randint(0, 1) for i in range(len(data))])
        # self.discard_items_over_capacity(data, capacity, item_ranking)
        self.discard_items_over_capacity2(data,capacity)

    def calculate_function_value(self, data):
        self.function_value = np.sum(self.solution * data[:, 2])

    def calculate_fitness(self):
        if self.function_value >= 0:
            self.fitness = 1 / (1 + self.function_value)
        else:
            self.fitness = 1 + abs(self.function_value)

    def calcutale_probability(self, all_probabilities):
        self.probability = self.function_value / all_probabilities

    def mix_solution(self, data, capacity, index):
        weights = data[:, 1]
        while True:
            current_weight = np.sum(self.solution * weights)
            tmp = random.choice(np.argwhere(self.solution == 0).reshape(-1))
            if tmp != index:
                if current_weight + weights[tmp] <= capacity:
                    self.solution[tmp] = 1
                else:
                    break

    def mix_solution2(self, data, capacity, index, item_ranking):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)
        ir = np.argwhere(item_ranking[:, 0] == index).reshape(-1)[0]
        # print(self.solution)
        for i in item_ranking[ir+1:]:
            if self.solution[int(i[0])] == 0:
                if current_weight + weights[int(i[0])] <= capacity:
                    self.solution[int(i[0])] = 1
                    current_weight += weights[int(i[0])]
        # print(self.solution)

    def mix_solution3(self, data, capacity, index, item_ranking):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)
        for i in item_ranking:
            if np.any(index == int(i[0])):
                continue
            else:
                if current_weight + weights[int(i[0])] <= capacity:
                    if self.solution[int(i[0])] == 0:
                        self.solution[int(i[0])] = 1
                        current_weight += weights[int(i[0])]

    def mix_solution4(self, data, capacity, item_ranking):
        weights = data[:, 1]
        current_weight = np.sum(self.solution * weights)

        j = 30
        while j:
            i = np.argwhere(self.solution == 0).reshape(-1)[0]
            if current_weight + weights[int(i)] <= capacity:
                self.solution[int(i)] = 1
                current_weight += weights[int(i)]
            if current_weight == capacity:
                break
            j = j - 1

        for index, probability in item_ranking:
            if self.solution[int(index)] == 0:
                if current_weight + weights[int(index)] <= capacity:
                    self.solution[int(index)] = 0
                    current_weight += weights[int(index)]
                    self.weight = current_weight

    def dopchaj_solution(self, data, capacity, item_ranking):
        weights = data[:, 1]
        current_weight = np.sum(weights * self.solution)
        for item, probability in item_ranking[::-1]:
            if self.solution[int(item)] == 0:
                if current_weight + weights[int(item)] <= capacity:
                    self.solution[int(item)] = 1
                    current_weight += weights[int(item)]
            if current_weight == capacity:
                break
        self.weight = current_weight

class Hive:
    def __init__(self, swarm_size, data, capacity):
        score = data[:, 2] / data[:, 1]
        probability_of_items = capacity / np.sum(data[:, 1]) * score / np.mean(score)
        probability_of_items /= (np.sum(probability_of_items))
        probability_of_items = np.concatenate((data[:, 0, np.newaxis], probability_of_items[:, np.newaxis]), axis=1)
        item_ranking = probability_of_items[probability_of_items[:, 1].argsort()]

        self._swarm_size = swarm_size
        self._item_ranking = item_ranking
        self._data = data
        self._capacity = capacity
        self.best_food_source = None
        self.best_function_value = 0
        self._hive = [Bee() for i in range(round(swarm_size / 2))]
        self.weight = 0

        # self._greed = self.init_greed_solution(self._data, self._capacity)

        for bee in self._hive:
            bee.set_random_solution(self._data, self._capacity, self._item_ranking)
            # bee.solution = self._greed
            bee.calculate_function_value(self._data)
            bee.calculate_fitness()

    def employed_bee_phase(self):
        for bee in self._hive:
            # index = np.empty(3)
            new_bee = copy.deepcopy(bee)
            index = random.choice(np.argwhere(new_bee.solution == 0).reshape(-1))
            new_bee.solution[index] = 1
            new_bee.discard_items_over_capacity(data=self._data, capacity=self._capacity,
                                                item_ranking=self._item_ranking)

            # new_bee.mix_solution2(data=self._data, capacity=self._capacity, index=index, item_ranking=self._item_ranking[::-1])
            # new_bee.discard_items_over_capacity2(self._data,self._capacity)
            new_bee.calculate_function_value(data=self._data)
            new_bee.calculate_fitness()
            if bee.fitness > new_bee.fitness:
                bee = new_bee
                bee.trial = 0
            else:
                bee.trial += 1
        # print("debug")

    def onlooker_bee_phase(self):
        function_value_sum = sum([bee.function_value for bee in self._hive])
        for bee in self._hive:
            bee.calcutale_probability(function_value_sum)

        number_of_onlooker_bees = int(self._swarm_size / 2)
        while number_of_onlooker_bees > 0:
            for bee in self._hive:
                if number_of_onlooker_bees < 1:
                    break
                r = random.uniform(0, 1)
                if r < bee.probability:
                    new_bee = copy.deepcopy(bee)
                    index = random.choice(np.argwhere(new_bee.solution == 0).reshape(-1))
                    new_bee.solution[index] = 1
                    new_bee.discard_items_over_capacity(data=self._data, capacity=self._capacity,
                                                        item_ranking=self._item_ranking)
                    # new_bee.mix_solution2(self._data, self._capacity, index, self._item_ranking[::-1])
                    # new_bee.discard_items_over_capacity2(self._data, self._capacity)
                    new_bee.calculate_function_value(data=self._data)
                    new_bee.calculate_fitness()
                    if bee.fitness > new_bee.fitness:
                        bee = new_bee
                        bee.trial = 0
                    else:
                        bee.trial += 1
                    number_of_onlooker_bees -= 1

        for bee in self._hive:
            if bee.function_value > self.best_function_value:
                self.best_function_value = bee.function_value
                self.best_food_source = bee.solution
                self.weight = bee.weight

    def scout_bee_phase(self, limit):
        for bee in self._hive:
            if bee.trial > limit:
                bee.set_random_solution(self._data, self._capacity, self._item_ranking)
                # bee.solution = self._greed
                bee.dopchaj_solution(self._data, self._capacity, self._item_ranking)

                bee.calculate_function_value(self._data)
                bee.calculate_fitness()

    def run(self, number_of_cycles, limit):
        for i in range(number_of_cycles):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase(limit)
            # print("BEST SO FAR:",self.best_function_value)
        print("Hello from bee")
        print(self.best_function_value)
        print(self.best_food_source)
        # print(data.T)
        # print("Best food source")
        # print(f" {self.best_food_source}")
        # print(f"with highest value in backpack: {self.best_function_value}")
        # print(f"with weight: {self.weight}")
        return self.best_function_value, np.count_nonzero(self.best_food_source)

    def init_greed_solution(self, data, C):
        score = data[:, 2] / data[:, 1]
        data = np.concatenate((data, score[:, np.newaxis]), axis=1)
        data = data[data[:, 3].argsort()][::-1]
        lim = np.shape(data)[0]
        i = 0
        backpack = []
        while i < lim:
            if data[i, 1] <= C:
                C = C - data[i, 1]
                backpack.append(data[i, :])
                i = i + 1
            else:
                i = i + 1
        np_b = np.array(backpack)
        food_source = np.linspace(0, 0, lim)
        for i in range(len(backpack)):
            food_source[int(np_b[i, 0])] = 1
        return food_source


def generate_data(amount, max_weight, max_value):
    random_weights = [random.randint(1, max_weight) for i in range(amount)]  # sample(range(1, max_weight), amount)
    random_values = [random.randint(1, max_value) for i in range(amount)]

    ids = np.linspace(0, amount - 1, amount).astype(int)
    weights = np.array(random_weights)
    values = np.array(random_values)

    return np.concatenate(
        (ids[:, np.newaxis], weights[:, np.newaxis], values[:, np.newaxis]), axis=1)


if __name__ == "__main__":
    number_of_items = 100
    data = generate_data(number_of_items, 10, 10)
    capacity = round(0.35 * np.round(np.sum(data[:, 1])))
    swarm_size = 200
    print(capacity)
    # print(data.T)

    ABC = Hive(swarm_size=swarm_size, data=data, capacity=capacity)
    print(ABC.run(number_of_cycles=100, limit=10))
