from scipy.stats import norm

# Convert times to “minutes after midnight”
mu_train    = 8*60 + 44   # 08:44 → 524 min
sigma_train = 3           # minutes
due_train   = 8*60 + 45   # 08:45 → 525 min

mu_bus      = 8*60 + 50   # 08:50 → 530 min
sigma_bus   = 1           # minutes

# 1) P(train is late) = P(X_train > 525)
p_train_late = 1 - norm.cdf(due_train, loc=mu_train, scale=sigma_train)

# 2) P(bus departs before train arrives) = P(X_train − Y_bus > 0)
mu_diff     = mu_train - mu_bus
sigma_diff  = (sigma_train**2 + sigma_bus**2)**0.5
p_bus_before_train = 1 - norm.cdf(0, loc=mu_diff, scale=sigma_diff)

print(f"P(train arrives after 08:45) = {p_train_late:.4f}")
print(f"P(bus leaves before train arrives) = {p_bus_before_train:.4f}")
