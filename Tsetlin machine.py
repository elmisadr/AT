import numpy as np
import random
import scipy.stats as stats
import math

class TsetlinMachine:
    def __init__(self, number_of_clauses, number_of_features, number_of_states, s, threshold):
        self.number_of_clauses = number_of_clauses
        self.number_of_features = number_of_features
        self.number_of_states = number_of_states
        self.s = s
        self.threshold = threshold

        self.ta_state = np.random.choice([self.number_of_states, self.number_of_states+1], size=(self.number_of_clauses, self.number_of_features, 2)).astype(dtype=np.int32)

        self.clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)
        
        self.clause_output = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)
        self.feedback_to_clauses = np.zeros(shape=(self.number_of_clauses), dtype=np.int32)

        for j in range(self.number_of_clauses):
            if j % 2 == 0:
                self.clause_sign[j] = 1
            else:
                self.clause_sign[j] = -1

    def calculate_clause_output(self, X):
        for j in range(self.number_of_clauses):
            self.clause_output[j] = 1
            for k in range(self.number_of_features):
                action_include = self.action(self.ta_state[j,k,0])
                action_include_negated = self.action(self.ta_state[j,k,1])

                if (action_include == 1 and X[k] == 0) or (action_include_negated == 1 and X[k] == 1):
                    self.clause_output[j] = 0
                    break

    def predict(self, X):
        self.calculate_clause_output(X)

        output_sum = self.sum_up_clause_votes()

        if output_sum >= 0:
            return 1
        else:
            return 0

    def action(self, state):
        if state <= self.number_of_states:
            return 0
        else:
            return 1

    def get_state(self, clause, feature, automaton_type):
        return self.ta_state[clause,feature,automaton_type]

    def sum_up_clause_votes(self):
        output_sum = 0
        for j in range(self.number_of_clauses):
            output_sum += self.clause_output[j]*self.clause_sign[j]

        if output_sum > self.threshold:
            output_sum = self.threshold

        elif output_sum < -self.threshold:
            output_sum = -self.threshold

        return output_sum

    def evaluate(self, X, y, number_of_examples):
        errors = 0
        for l in range(number_of_examples):
            Xi = X[l]

            self.calculate_clause_output(Xi)

            output_sum = self.sum_up_clause_votes()

            if output_sum >= 0 and y[l] == 0:
                errors += 1

            elif output_sum < 0 and y[l] == 1:
                errors += 1

        return 1.0 - 1.0 * errors / number_of_examples
    

    def update(self, X, y,alpha):
        self.calculate_clause_output(X)
        output_sum = self.sum_up_clause_votes()

        for j in range(self.number_of_clauses):
            self.feedback_to_clauses[j] = 0

        if y == 1:
        # Calculate feedback to clauses
            for j in range(self.number_of_clauses):
                if 1.0 * random.random() > 1.0 * (self.threshold - output_sum) /(2 * self.threshold):
                    continue

                if self.clause_sign[j] >= 0:
                # Type I Feedback
                   self.feedback_to_clauses[j] = 1
                else:
                # Type II Feedback
                   self.feedback_to_clauses[j] = -1

        elif y == 0:
            for j in range(self.number_of_clauses):
                if 1.0 * random.random() > 1.0 * (self.threshold + output_sum) / (2 * self.threshold):
                    continue

                if self.clause_sign[j] >= 0:
                # Type I Feedback
                   self.feedback_to_clauses[j] = 1
                else:
                # Type II Feedback
                   self.feedback_to_clauses[j] = -1

        for j in range(self.number_of_clauses):
            if self.feedback_to_clauses[j] > 0:
          
 ### Modified Type I Feedback (New Tables)   
                initial_stddev = 0.1
                decay_rate = 0.01
                for epoch in range(epochs):

                    sigma_e = initial_stddev * np.exp(-decay_rate * epoch)
                 
                    def calculate_equation(s, d, epoch):
                        argument = -(s - 2) / s / math.sqrt(2 * math.exp(-2 * decay_rate * epoch))
                        alpha = 1 - stats. norm.cdf(argument)
                        return alpha

                    if self.clause_output[j] == 0:
                    
                        for k in range(self.number_of_features):
                            if X[k] == 1:
                                if alpha > 0.5:
                                        if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                            self.ta_state[j, k, 1] -= math.floor((self.s-1)*(1-alpha)/alpha)
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] -= (self.s-1)
        
                                elif alpha < 0.5:
                                        if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                            self.ta_state[j, k, 1] += (self.s-1)
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] += math.floor((self.s-1)*alpha/(1-alpha))

                            elif X[k] == 0:
                                if alpha > 0.5:
                                        if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                            self.ta_state[j, k, 1] -= math.floor((self.s-1)*(1-alpha)/alpha)
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] -= (self.s-1)
        
                                elif alpha<=0.5:
                                        if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                            self.ta_state[j, k, 1] += (self.s-1)
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] +=math.floor((self.s-1)*alpha/(1-alpha))

                    if self.clause_output[j] == 1:
                            for k in range(self.number_of_features):
                                if X[k] == 1:
                                    if alpha > 0.5:
                                        if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                            self.ta_state[j, k, 1] += (self.s-1)
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] += math.floor((self.s-1)*(1-alpha)/alpha)
        
                                    elif alpha<=0.5:
                                        if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                            self.ta_state[j, k, 1] -= math.floor((self.s-1)*alpha/(1-alpha))
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] -= (self.s-1)

 
                                elif X[k] == 0:
                                    if alpha > 0.5:
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] -= (self.s-1)
                                        
                                    elif alpha<=0.5:
                                        if self.ta_state[j, k, 0] > 1:
                                            self.ta_state[j, k, 0] +=math.floor((self.s-1)*alpha/(1-alpha))
 
            elif self.feedback_to_clauses[j] < 0:                             
                                        
          # Type II Feedback (Combats False Positive Output)
                    if self.clause_output[j] == 1:
                            for k in range(self.number_of_features):
                                    action_include = self.action(self.ta_state[j,k,0])
                                    action_include_negated = self.action(self.ta_state[j,k,1])

                                    if X[k] == 0:
                                            if action_include == 0 and self.ta_state[j,k,0] < self.number_of_states*2:
                                                    self.ta_state[j,k,0] += 1
                                    elif X[k] == 1:
                                           if action_include_negated == 0 and self.ta_state[j,k,1] < self.number_of_states*2:
                                                   self.ta_state[j,k,1] += 1


    def fit(self, X: np.ndarray, y: np.ndarray, number_of_examples: int, epochs: int):
        for epoch in range(epochs):

            np.random.shuffle(X)
            np.random.shuffle(y)
            for example_id in range(number_of_examples):
                target_class = y[example_id]

                Xi = X[example_id, :].astype(np.int32)
                self.update(Xi, target_class)
               
                
