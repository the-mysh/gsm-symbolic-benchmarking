# packages needed for pymer4 Python package to work on R 3.6.3
# call with: `Rscript setup.R`
install.packages("remotes")

# Core dependencies
remotes::install_version("rlang", version = "0.4.12")
remotes::install_version("vctrs", version = "0.3.8")
remotes::install_version("lifecycle", version = "1.0.1")
remotes::install_version("pillar", version = "1.6.4")
remotes::install_version("tibble", version = "3.1.6")

# Matrix and lme4
install.packages("Matrix")
install.packages("lme4")

# tidyverse dependencies
remotes::install_version("purrr", version = "0.3.5")
remotes::install_version("tidyselect", version = "1.1.0")
remotes::install_version("dplyr", version = "1.0.3")
remotes::install_version("cpp11", version = "0.4.3")
install.packages("tidyr")
remotes::install_version("stringr", version = "1.4.1")
remotes::install_version("gtable", version = "0.3.1")
remotes::install_version("scales", version = "1.1.1")
remotes::install_version("ggplot2", version = "3.3.6")
remotes::install_version("forcats", version = "0.5.1")
remotes::install_version("furrr", version = "0.2.3")

# broom
remotes::install_version("broom", version = "0.7.12")
remotes::install_version("broom.mixed", version = "0.2.9.4")

# lmerTest
install.packages("lmerTest")

# emmeans
remotes::install_version("estimability", version = "1.4.1")
remotes::install_version("emmeans", version = "1.7.5")

# report and its dependencies
remotes::install_version("datawizard", version = "0.6.5")
remotes::install_version("bayestestR", version = "0.12.1")
remotes::install_version("parameters", version = "0.18.2")
remotes::install_version("performance", version = "0.9.2")
remotes::install_version("effectsize", version = "0.7.0")
remotes::install_version("report", version = "0.5.5")