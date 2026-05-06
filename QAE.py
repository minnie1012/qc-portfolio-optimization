import numpy as np
# 1. Use the modern V2 Primitive 
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import IterativeAmplitudeEstimation
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications.estimation import EuropeanCallPricing

# ---------------------------------------------------------
# 1. Financial Setup (4 Qubits for higher resolution)
# ---------------------------------------------------------
num_uncertainty_qubits = 4  

spot_price = 100.0      
volatility = 0.4        
risk_free_rate = 0.05   
maturity = 40 / 365     
strike_price = 110.0    

# Log-Normal Math
mu = (risk_free_rate - 0.5 * volatility**2) * maturity + np.log(spot_price)
sigma = volatility * np.sqrt(maturity)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

low  = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, 
    mu=mu, 
    sigma=sigma**2, 
    bounds=(low, high)
)

# ---------------------------------------------------------
# 2. The Native Qiskit Finance Wrapper
# ---------------------------------------------------------
# This automatically handles the complex baseline math and scaling
# so the amplitude never gets misinterpreted as a negative dollar amount.
european_call = EuropeanCallPricing(
    num_state_qubits=num_uncertainty_qubits,
    strike_price=strike_price,
    rescaling_factor=0.05,  
    bounds=(low, high),
    uncertainty_model=uncertainty_model
)

problem = european_call.to_estimation_problem()

# ---------------------------------------------------------
# 3. Execute High-Precision IQAE (Modern Framework)
# ---------------------------------------------------------
# By default, StatevectorSampler computes the exact mathematical 
# probabilities without introducing shot noise.
sampler = StatevectorSampler()

# TIGHTENED EPSILON: 0.001 forces the algorithm to be highly precise
iqae = IterativeAmplitudeEstimation(epsilon_target=0.001, alpha=0.05, sampler=sampler)

print("Running Modern High-Precision IQAE (V2 Sampler)...")
result = iqae.estimate(problem)

# ---------------------------------------------------------
# 4. Output Results
# ---------------------------------------------------------
print("\n--- Final Results ---")
print(f"Estimated Expected Payoff: ${result.estimation_processed:.4f}")
print(f"95% Confidence Interval:   [${result.confidence_interval_processed[0]:.4f}, ${result.confidence_interval_processed[1]:.4f}]")
print(f"Total Oracle Queries (k):  {result.num_oracle_queries}")