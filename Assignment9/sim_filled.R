f <- function(x) x**2

require("grDevices") # for colours
filled.contour(volcano, asp = 1) # simple



x <- seq(0, 1, 0.01)
y <- x*x
s = smooth.spline(x, y, spar=0.5)
xy <- predict(s, seq(min(x), max(x), by=0.01)) # Some vertices on the curve
m <- length(xy$x)                         
x.poly <- c(xy$x, xy$x[m], xy$x[1])         # Adjoin two x-coordinates
y.poly <- c(xy$y, 0, 0)                     # .. and the corresponding y-coordinates
plot(range(x), c(0, max(y)), type='n', xlab="X", ylab="Y")
polygon(x.poly, y.poly, col="orange", border=NA)          # Show the polygon fill only
lines(s)

f <- runif(200)
plot(f, xlab = "x", ylab = "Random")



# Load the necessary library
library(ggplot2)

# Define the function and the range of x
x <- seq(0, 1, length.out = 100)
y <- x^2

# Create a data frame with x and y
data <- data.frame(x, y)

# Plot using ggplot2 and fill the area under the curve
ggplot(data, aes(x = x, y = y)) +
  geom_line() +
  geom_ribbon(aes(ymin = 0, ymax = y), fill = "blue", alpha = 0.3) +
  labs(title = "Area under the curve y = x^2",
       x = "x",
       y = "y") +
  theme_minimal()


# Load the necessary library
library(ggplot2)

# Define the function and the range of x
x <- seq(0, 1, length.out = 100)
y <- x^2

# Generate random uniform values
f <- runif(200)

# Create a data frame with x and y
data <- data.frame(x, y)

# Create a data frame for the random uniform values
# Since f has 200 values, we'll need to adjust x to match this length
x_f <- seq(0, 1, length.out = 200)
data_f <- data.frame(x = x_f, y = f)

# Plot using ggplot2 and fill the area under the curve
ggplot() +
  # Plot the y = x^2 curve and fill the area
  geom_line(data = data, aes(x = x, y = y)) +
  geom_ribbon(data = data, aes(x = x, ymin = 0, ymax = y), fill = "blue", alpha = 0.3) +
  # Overlay the random uniform points
  geom_point(data = data_f, aes(x = x, y = y), color = "red", size = 1) +
  annotate("text", x = 0.5, y = 0.5, label = "y = x^2", color = "black", size = 5, angle = 0) +
  labs(title = "Area under the curve y = x^2 with random uniform points",
       x = "x",
       y = "y") +
  theme_minimal()


