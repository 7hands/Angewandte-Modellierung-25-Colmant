# Convert times to “minutes after midnight”
mu_train    <- 8*60 + 44  # 524
sigma_train <- 3
due_train   <- 8*60 + 45  # 525

mu_bus      <- 8*60 + 50  # 530
sigma_bus   <- 1

# 1) P(train is late)
p_train_late <- 1 - pnorm(due_train, mean = mu_train, sd = sigma_train)

# 2) P(bus leaves before train arrives)
mu_diff      <- mu_train - mu_bus
sigma_diff   <- sqrt(sigma_train^2 + sigma_bus^2)
p_bus_before_train <- 1 - pnorm(0, mean = mu_diff, sd = sigma_diff)

cat("P(train arrives after 08:45) =", round(p_train_late, 4), "\n")
cat("P(bus leaves before train arrives) =", round(p_bus_before_train, 4), "\n")
