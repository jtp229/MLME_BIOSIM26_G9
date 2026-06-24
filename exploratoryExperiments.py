import basic_client


client = basic_client.BioreactorClient(basic_client.BASE_URL)
client.login(basic_client.USER, basic_client.PASSWORD)

#micro, pilot, bench
#result = client.run("bench", T=30.0, pH=6.5, F1=0.5, F2=0.5, F3=0.5)
#print(result)

result = client.run("bench", T=30.0, pH=6.5, F1=0.5, F2=0.5, F3=0.5)




