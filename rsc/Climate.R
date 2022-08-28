wd <- "D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/dat/clm/use"

db_fig_dir <- "C:/Users/Manuel/Dropbox/Apps/Overleaf/Masterarbeit/fig"
D_fig_dir <- "D:/Dateien/Studium_KIT/Master_GOEK/Masterarbeit/fig"

require("lubridate")
T_Marken <- read.csv(file.path(wd, "Temperature_marken.csv"))
T_Marken$Date <- as.Date(T_Marken$Date)
T_Marken$DoY <- yday(T_Marken$Date)
T_Marken$DoY_S <- T_Marken$DoY + 183
T_Marken$DoY_S[which(T_Marken$DoY_S > 365)] <- T_Marken$DoY_S[which(T_Marken$DoY_S > 365)] - 365

p_Frishgewaagd <- read.csv(file.path(wd, "Rainfall_Frishgewaagd.csv"))
p_Frishgewaagd$Date <- paste(p_Frishgewaagd$Year, match(p_Frishgewaagd$Month, month.name), "15", sep = "-")
p_Frishgewaagd$Date <- as.Date(p_Frishgewaagd$Date)
p_Frishgewaagd <- p_Frishgewaagd[which(p_Frishgewaagd$Date >= as.Date("1990-07-15")),]
p_Frishgewaagd$Month <- ordered(p_Frishgewaagd$Month, levels = p_Frishgewaagd$Month[seq(1, 12)])

# summarise data
se <- function(x, na.rm = TRUE){
  if(na.rm){
    x_0 <- x[which(!is.na(x))]
  }else{
    x_0 <- x
  }
  return(sd(x_0) / sqrt(length(x_0)))
}

library("dplyr")
p_F_summary <- p_Frishgewaagd %>%
  group_by(Month) %>%
  summarise(average = mean(p_in_mm, na.rm = TRUE), se = se(p_in_mm, na.rm = TRUE))
p_F_summary$DoY_S <- yday(paste(1900, match(p_F_summary$Month, month.name), "15", sep = "-"))
p_F_summary$DoY_S <- p_F_summary$DoY_S + 183
p_F_summary$DoY_S[which(p_F_summary$DoY_S > 365)] <- p_F_summary$DoY_S[which(p_F_summary$DoY_S > 365)] - 365
p_F_summary$M <- substr(p_F_summary$Month, 1, 1)

T_M_summary <- T_Marken %>%
  group_by(DoY_S, Variable) %>%
  summarise(average = mean(Temperature, na.rm = TRUE))

T_M_summary_month <- T_Marken %>%
  group_by(Month, Variable) %>%
  summarise(average = mean(Temperature, na.rm = TRUE))

# plot
library("ggplot2")
theme_set(theme_bw())
cols <- c(
  rgb(0, 150, 130, alpha = 254.99, maxColorValue = 255), #kit colour
  rgb(70, 100, 170, alpha = 254.99, maxColorValue = 255), #kit blue
  rgb(223, 155, 27, alpha = 254.99, maxColorValue = 255), #kit orange
  rgb(140, 182, 60, alpha = 254.99, maxColorValue = 255), #kit Mai green
  rgb(162, 34, 35, alpha = 254.99, maxColorValue = 255), #kit red
  rgb(163, 16, 124, alpha = 254.99, maxColorValue = 255), #kit violet
  rgb(167, 130, 46, alpha = 254.99, maxColorValue = 255), #kit brown
  rgb(252, 229, 0, alpha = 254.99, maxColorValue = 255), #kit yellow
  rgb(25, 161, 224, alpha = 254.99, maxColorValue = 255)# kit cyan
)

# scale values for simultaneous display of T and p
p_F_summary$avr <- p_F_summary$average / 2
p_F_summary$ase <- p_F_summary$se / 2
T_M_summary$avr <- T_M_summary$average
# plot
gg_clim <- ggplot(data = T_M_summary, aes(x = DoY_S, y = avr)) +
  geom_bar(data = p_F_summary, aes(y = avr), stat = "identity", fill = cols[2]) +
  geom_errorbar(data = p_F_summary, aes(ymin = avr-ase, ymax = avr+ase),
                width = 10, colour = "black", alpha = 1, size = 0.5) +
  geom_smooth(aes(colour = Variable), se = TRUE) +
  scale_x_continuous(name = element_blank(),
                     breaks = p_F_summary$DoY_S,
                     labels = p_F_summary$M) +
  scale_y_continuous(name = "Temperature [°C]", sec.axis = sec_axis(~ . * 2, name = "Precipitation [mm]")) +
  theme(legend.position = "none") +
  scale_color_discrete(cols)

# check by adding monthly average temperatures
T_M_summary_month$DoY_S <- rep(p_F_summary$DoY_S, each = 2) + 183
T_M_summary_month$DoY_S[which(T_M_summary_month$DoY_S > 365)] <- T_M_summary_month$DoY_S[which(T_M_summary_month$DoY_S > 365)] - 365
T_M_summary_month$avr <- T_M_summary_month$average
gg_clim +
  geom_point(data = T_M_summary_month, aes(colour = Variable))

# create pdf in page width
A4_width <- 8.25
margin_left <- 2.3 * 0.393701
margin_right <- 2.3 * 0.393701
pagewidth <- A4_width - margin_left - margin_right

pdf(file = file.path(db_fig_dir, "Climate.pdf"), width = pagewidth, height = pagewidth/4*3)
print(gg_clim)
dev.off()