# Getting Started

### Reference Documentation
Gradle Official Documentation

* [Official Gradle documentation](https://docs.gradle.org)
* [Spring Boot Gradle Plugin Reference Guide](https://docs.spring.io/spring-boot/docs/3.1.5/gradle-plugin/reference/html/)
* [Spring Boot DevTools](https://docs.spring.io/spring-boot/docs/3.1.5/reference/htmlsingle/index.html#using.devtools)
* [Spring Reactive Web](https://docs.spring.io/spring-boot/docs/3.1.5/reference/htmlsingle/index.html#web.reactive)



# Classifying Pingu with a Neural Network

Using the provided weightings for a neural network with 4 input nodes, 2 hidden layer nodes and 3 output nodes
a single feed forward pass was executed against the first data instance in the penguins training dataset. 
This initial row had a classifier response variable identifying the observation as an Adelie penguin, 
and was classified incorrectly by the neural network with the initial weights as a Gentoo. 
This was consistent when no initial bias weights were used and when bias nodes were used.

> java -jar a2p1-nb.jar
OL outputs: [0.47627171854273365, 0.5212308085921598, 0.47617240962812674]
Predicted label for the first instance is: Gentoo

> java -jar a2p1-wb.jar
bias nodes: [-0.02, -0.2, -0.33, 0.26, 0.06]
OL outputs: [0.39471646725685955, 0.5837870023385486, 0.49538270744527696]
Predicted label for the first instance is: Gentoo

After 100 epochs there is an observable difference in the final weights between the run with no-bias applied and the run with bias. 


```
------------------------------
 Feed forward pass
------------------------------
inputs: [0.15636363636363626, 0.48148148148148157, 0.34545454545454546, 0.1865671641791045]
bias nodes: [-0.02, -0.2, -0.33, 0.26, 0.06]
Z5=(-28.000000 * 0.156364) + (0.080000 * 0.481481) + (-0.300000 * 0.345455) + (0.100000 * 0.186567) + -0.020000 = -4.444643
O5=0.011605039329659693
Z6=(-0.220000 * 0.156364) + (0.200000 * 0.481481) + (0.320000 * 0.345455) + (0.010000 * 0.186567) + -0.200000 = -0.025693
O6=0.4935772089286013
Z7=(-0.290000 * 0.011605) + (0.080000 * 0.493577) + -0.330000 = -0.293879
O7=0.42705442031231794
Z8=(0.030000 * 0.011605) + (0.130000 * 0.493577) + 0.260000 = 0.324513
O8=0.5804237552377884
Z9=(0.210000 * 0.011605) + (-0.360000 * 0.493577) + 0.060000 = -0.115251
O9=0.4712191660538079
HL outputs: [0.011605039329659693, 0.4935772089286013]
OL outputs: [0.42705442031231794, 0.5804237552377884, 0.4712191660538079]
Predicted label for the first instance is: Gentoo
------------------------------
 Feed forward pass
------------------------------
inputs: [0.15636363636363626, 0.48148148148148157, 0.34545454545454546, 0.1865671641791045]
bias nodes: [-0.02, -0.2, -0.33, 0.26, 0.06]
Z5=(-28.000000 * 0.156364) + (0.080000 * 0.481481) + (-0.300000 * 0.345455) + (0.100000 * 0.186567) + -0.020000 = -4.444643
O5=0.011605039329659693
Z6=(-0.220000 * 0.156364) + (0.200000 * 0.481481) + (0.320000 * 0.345455) + (0.010000 * 0.186567) + -0.200000 = -0.025693
O6=0.4935772089286013
Z7=(-0.290000 * 0.011605) + (0.080000 * 0.493577) + -0.330000 = -0.293879
O7=0.42705442031231794
Z8=(0.030000 * 0.011605) + (0.130000 * 0.493577) + 0.260000 = 0.324513
O8=0.5804237552377884
Z9=(0.210000 * 0.011605) + (-0.360000 * 0.493577) + 0.060000 = -0.115251
O9=0.4712191660538079
HL outputs: [0.011605039329659693, 0.4935772089286013]
OL outputs: [0.42705442031231794, 0.5804237552377884, 0.4712191660538079]
------------------------------
 Back Propagation
------------------------------
b7=0-0.42705442031231794 => -0.42705442031231794
b8=1-0.5804237552377884 => 0.41957624476221156
b9=0-0.4712191660538079 => -0.4712191660538079
OL betas: [-0.42705442031231794, 0.41957624476221156, -0.4712191660538079]
b5=(-0.290000 * 0.427054 * (1 - 0.427054) * -0.427054) +
(0.030000 * 0.580424 * (1 - 0.580424) * 0.419576) +
(0.210000 * 0.471219 * (1 - 0.471219) * -0.471219) +
 --> 0.00871082510183284
b6=(0.080000 * 0.427054 * (1 - 0.427054) * -0.427054) +
(0.130000 * 0.580424 * (1 - 0.580424) * 0.419576) +
(-0.360000 * 0.471219 * (1 - 0.471219) * -0.471219) +
 --> 0.047193341489822446
HL betas: [0.00871082510183284, 0.047193341489822446]
w57 = 0.200000 * 0.011605 * 0.427054 * (1 - 0.427054) * -0.427054
w58 = 0.200000 * 0.011605 * 0.580424 * (1 - 0.580424) * 0.419576
w59 = 0.200000 * 0.011605 * 0.471219 * (1 - 0.471219) * -0.471219
w67 = 0.200000 * 0.493577 * 0.427054 * (1 - 0.427054) * -0.427054
w68 = 0.200000 * 0.493577 * 0.580424 * (1 - 0.580424) * 0.419576
w69 = 0.200000 * 0.493577 * 0.471219 * (1 - 0.471219) * -0.471219
Delta output weights: [[-2.425249526182475E-4, 2.3716116459723146E-4, -2.725198934147492E-4], [-0.010314897331103327, 0.010086768546228676, -0.011590620638863118]]
w15 = 0.200000 * 0.156364 * 0.011605 * (1 - 0.011605) * 0.008711
w16 = 0.200000 * 0.156364 * 0.493577 * (1 - 0.493577) * 0.047193
w25 = 0.200000 * 0.481481 * 0.011605 * (1 - 0.011605) * 0.008711
w26 = 0.200000 * 0.481481 * 0.493577 * (1 - 0.493577) * 0.047193
w35 = 0.200000 * 0.345455 * 0.011605 * (1 - 0.011605) * 0.008711
w36 = 0.200000 * 0.345455 * 0.493577 * (1 - 0.493577) * 0.047193
w45 = 0.200000 * 0.186567 * 0.011605 * (1 - 0.011605) * 0.008711
w46 = 0.200000 * 0.186567 * 0.493577 * (1 - 0.493577) * 0.047193
bias5 = 0.200000 * 0.011605 * (1 - 0.011605) * 0.008711
bias6 = 0.200000 * 0.493577 * (1 - 0.493577) * 0.047193
bias7 = 0.200000 * 0.427054 * (1 - 0.427054) * -0.427054
bias8 = 0.200000 * 0.580424 * (1 - 0.580424) * 0.419576
bias9 = 0.200000 * 0.471219 * (1 - 0.471219) * -0.471219
Bias deltas: [1.998326412995133E-5, 0.002359277708232635][-0.020898244782196648, 0.02043605005207561, -0.023482892704917754]
Delta hidden weights: [[3.124655845774206E-6, 3.689052416509209E-4], [9.621571618124718E-6, 0.001135948526186084], [6.90330942671046E-6, 8.150232082985467E-4], [3.72822091976704E-6, 4.401637515359394E-4]]
Hidden layer weights 
[[-27.999996875344156, -0.21963109475834908], [0.08000962157161813, 0.2011359485261861], [-0.2999930966905733, 0.3208150232082986], [0.10000372822091977, 0.01044016375153594]]
Output layer weights  
[[-0.2902425249526182, 0.03023716116459723, 0.20972748010658523], [0.06968510266889667, 0.14008676854622867, -0.3715906206388631]]
accuracy predicted: [1/1]: epoch 0: accuracy = 1.000000
Weights after performing BP for first instance only:
Hidden layer weights:
[[-27.999996875344156, -0.21963109475834908], [0.08000962157161813, 0.2011359485261861], [-0.2999930966905733, 0.3208150232082986], [0.10000372822091977, 0.01044016375153594]]
Output layer weights:
[[-0.2902425249526182, 0.03023716116459723, 0.20972748010658523], [0.06968510266889667, 0.14008676854622867, -0.3715906206388631]]
accuracy predicted: [167/268]: epoch 0: accuracy = 0.623134
accuracy predicted: [153/268]: epoch 1: accuracy = 0.570896
accuracy predicted: [152/268]: epoch 2: accuracy = 0.567164
accuracy predicted: [150/268]: epoch 3: accuracy = 0.559701
accuracy predicted: [150/268]: epoch 4: accuracy = 0.559701
accuracy predicted: [151/268]: epoch 5: accuracy = 0.563433
accuracy predicted: [151/268]: epoch 6: accuracy = 0.563433
accuracy predicted: [153/268]: epoch 7: accuracy = 0.570896
accuracy predicted: [154/268]: epoch 8: accuracy = 0.574627
accuracy predicted: [161/268]: epoch 9: accuracy = 0.600746
accuracy predicted: [167/268]: epoch 10: accuracy = 0.623134
accuracy predicted: [176/268]: epoch 11: accuracy = 0.656716
accuracy predicted: [188/268]: epoch 12: accuracy = 0.701493
accuracy predicted: [197/268]: epoch 13: accuracy = 0.735075
accuracy predicted: [206/268]: epoch 14: accuracy = 0.768657
accuracy predicted: [210/268]: epoch 15: accuracy = 0.783582
accuracy predicted: [212/268]: epoch 16: accuracy = 0.791045
accuracy predicted: [214/268]: epoch 17: accuracy = 0.798507
accuracy predicted: [213/268]: epoch 18: accuracy = 0.794776
accuracy predicted: [213/268]: epoch 19: accuracy = 0.794776
accuracy predicted: [214/268]: epoch 20: accuracy = 0.798507
accuracy predicted: [215/268]: epoch 21: accuracy = 0.802239
accuracy predicted: [215/268]: epoch 22: accuracy = 0.802239
accuracy predicted: [215/268]: epoch 23: accuracy = 0.802239
accuracy predicted: [217/268]: epoch 24: accuracy = 0.809701
accuracy predicted: [217/268]: epoch 25: accuracy = 0.809701
accuracy predicted: [218/268]: epoch 26: accuracy = 0.813433
accuracy predicted: [218/268]: epoch 27: accuracy = 0.813433
accuracy predicted: [218/268]: epoch 28: accuracy = 0.813433
accuracy predicted: [218/268]: epoch 29: accuracy = 0.813433
accuracy predicted: [218/268]: epoch 30: accuracy = 0.813433
accuracy predicted: [218/268]: epoch 31: accuracy = 0.813433
accuracy predicted: [220/268]: epoch 32: accuracy = 0.820896
accuracy predicted: [221/268]: epoch 33: accuracy = 0.824627
accuracy predicted: [221/268]: epoch 34: accuracy = 0.824627
accuracy predicted: [222/268]: epoch 35: accuracy = 0.828358
accuracy predicted: [222/268]: epoch 36: accuracy = 0.828358
accuracy predicted: [223/268]: epoch 37: accuracy = 0.832090
accuracy predicted: [224/268]: epoch 38: accuracy = 0.835821
accuracy predicted: [224/268]: epoch 39: accuracy = 0.835821
accuracy predicted: [225/268]: epoch 40: accuracy = 0.839552
accuracy predicted: [226/268]: epoch 41: accuracy = 0.843284
accuracy predicted: [227/268]: epoch 42: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 43: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 44: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 45: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 46: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 47: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 48: accuracy = 0.847015
accuracy predicted: [227/268]: epoch 49: accuracy = 0.847015
accuracy predicted: [229/268]: epoch 50: accuracy = 0.854478
accuracy predicted: [230/268]: epoch 51: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 52: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 53: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 54: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 55: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 56: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 57: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 58: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 59: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 60: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 61: accuracy = 0.858209
accuracy predicted: [230/268]: epoch 62: accuracy = 0.858209
accuracy predicted: [231/268]: epoch 63: accuracy = 0.861940
accuracy predicted: [231/268]: epoch 64: accuracy = 0.861940
accuracy predicted: [231/268]: epoch 65: accuracy = 0.861940
accuracy predicted: [231/268]: epoch 66: accuracy = 0.861940
accuracy predicted: [231/268]: epoch 67: accuracy = 0.861940
accuracy predicted: [232/268]: epoch 68: accuracy = 0.865672
accuracy predicted: [233/268]: epoch 69: accuracy = 0.869403
accuracy predicted: [233/268]: epoch 70: accuracy = 0.869403
accuracy predicted: [234/268]: epoch 71: accuracy = 0.873134
accuracy predicted: [235/268]: epoch 72: accuracy = 0.876866
accuracy predicted: [235/268]: epoch 73: accuracy = 0.876866
accuracy predicted: [235/268]: epoch 74: accuracy = 0.876866
accuracy predicted: [236/268]: epoch 75: accuracy = 0.880597
accuracy predicted: [236/268]: epoch 76: accuracy = 0.880597
accuracy predicted: [236/268]: epoch 77: accuracy = 0.880597
accuracy predicted: [236/268]: epoch 78: accuracy = 0.880597
accuracy predicted: [236/268]: epoch 79: accuracy = 0.880597
accuracy predicted: [237/268]: epoch 80: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 81: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 82: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 83: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 84: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 85: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 86: accuracy = 0.884328
accuracy predicted: [238/268]: epoch 87: accuracy = 0.888060
accuracy predicted: [237/268]: epoch 88: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 89: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 90: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 91: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 92: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 93: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 94: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 95: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 96: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 97: accuracy = 0.884328
accuracy predicted: [237/268]: epoch 98: accuracy = 0.884328
accuracy predicted: [238/268]: epoch 99: accuracy = 0.888060
Hidden layer weights:
[[-27.984504129320886, -6.1578240815692995], [0.15516205303161293, 6.281887857122405], [-0.26719582823686566, -2.9742492250893906], [0.12638271010701213, -0.6754443821009162]]
Output layer weights:
[[0.3647171531038541, -0.1371696785037134, -0.5651258682325535], [9.683643181901633, -12.22413045708797, -0.9161767583093584]]
accuracy predicted: [60/65]: Classify: testset [65] rows: accuracy = 0.923077

```