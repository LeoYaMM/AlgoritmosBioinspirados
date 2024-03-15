import math
import random

# Function to calculate number of bits needed for each decision variable
def calculate_num_bits(lower_bound, upper_bound, num_decimals):
    return math.ceil(math.log2((upper_bound * (10 ** num_decimals)) - (lower_bound * (10 ** num_decimals))))

# Calculate chromosome size for Rosenbrock and Ackley functions
rosenbrock_bits = calculate_num_bits(-2048, 2048, 2)
ackley_bits = calculate_num_bits(-32768, 32768, 2)

# Chromosome length for each variable (since we have 10 variables for each function)
chromosome_length_rosen = rosenbrock_bits * 10
chromosome_length_ack = ackley_bits * 10

# Encoding function: Convert a float value to a binary string with fixed size
def encode(value, lower_bound, upper_bound, chromosome_length):
    num_decimals = 2
    # Scale the float value to an integer value based on the number of decimals
    int_value = int((value - lower_bound) * (10 ** num_decimals))
    # Convert the integer to a binary string
    binary_string = format(int_value, '0' + str(chromosome_length) + 'b')
    return binary_string

# Decoding function: Convert a binary string to a float value
def decode(binary_string, lower_bound, upper_bound, chromosome_length):
    num_decimals = 2
    # Convert binary string to integer
    int_value = int(binary_string, 2)
    # Scale back the integer to a float value based on the number of decimals
    float_value = lower_bound + int_value / (10 ** num_decimals)
    return float_value

# Mutation function: Flip a bit in the chromosome with a certain probability
def mutate(chromosome, mutation_rate):
    mutated_chromosome = ''
    for bit in chromosome:
        if random.random() < mutation_rate:
            mutated_bit = '1' if bit == '0' else '0'
        else:
            mutated_bit = bit
        mutated_chromosome += mutated_bit
    return mutated_chromosome

# Test the encoding and decoding functions
test_value = 1.23 # example value
encoded_value = encode(test_value, -2.048, 2.048, rosenbrock_bits)
decoded_value = decode(encoded_value, -2.048, 2.048, rosenbrock_bits)

# Test the mutation function
original_chromosome = '0001' # example chromosome
mutation_rate = 0.01 # example mutation rate
mutated_chromosome = mutate(original_chromosome, mutation_rate)

encoded_value, decoded_value, original_chromosome, mutated_chromosome

# Print the chromosome sizes for both functions
print("Chromosome length for Rosenbrock function (per variable):", rosenbrock_bits)
print("Total chromosome length for Rosenbrock function:", chromosome_length_rosen)
print("Chromosome length for Ackley function (per variable):", ackley_bits)
print("Total chromosome length for Ackley function:", chromosome_length_ack)

# Print the test results for encoding and decoding
print("Test value to encode:", test_value)
print("Encoded binary string:", encoded_value)
print("Decoded value:", decoded_value)

# Print the test results for mutation
print("Original chromosome:", original_chromosome)
print("Mutation rate:", mutation_rate)
print("Mutated chromosome:", mutated_chromosome)
