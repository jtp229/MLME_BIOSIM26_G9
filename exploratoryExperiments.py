import basic_client
import os
import matplotlib.pyplot as plt
import csv

#Creates client object that acts as the API 
client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

#Hard coded bounds for each of the variables in the reactor
bounds = {"T": [20.,60.], "pH": [3., 9.5], "F1": [0.,2.], "F2": [0.,2.], "F3": [0.,2.]}
#List that the results are put into
result = []

#Lists for plot
xPoints = []
yPoints = []

#Got through each bound and add the value to xPoints for plotting, and run experiment in the reactor
for i in range(int(bounds["T"][0]), int(bounds["T"][1])):
    xPoints.append(i)

    result.append(client.run("bench", T=30.0, pH=6.5, F1=1, F2=1, F3=1))

    basic_client.time.sleep(0.5)



#Takes out Y value from dictionary and points it into yPoint list
for dict in result:
    yPoints.append(dict["Y"])

#Write values to csv
csv_filename = "bioreactor_data.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(["x_Time_or_Temp", "y_Yield"])
    
    # Zip xPoints and yPoints together and write row by row
    for x, y in zip(xPoints, yPoints):
        writer.writerow([x, y])


plt.plot(xPoints, yPoints)
plt.show()  







