from itertools import islice

text1 = ""
text2 = ""
text3 = ""
text4 = ""

with open("iphone_dev_token.txt", "r") as f:
    while True:
        line = list(islice(f, 1))
        if not line:
            break
        line = line[0].split("# # #")

        brand = line[0]
        brand = brand.strip()
        if (brand == "1"):
            text1 = text1 + line[1]
        elif (brand == "2"):
            text2 = text2 + line[1]
        elif (brand == "3"):
            text3 = text3 + line[1]
        elif (brand == "4"):
            text4 = text4 + line[1]
f.close()

f1 = open("positiveTest.txt", "w")
f2 = open("negativeTest.txt", "w")
f3 = open("neutralTest.txt", "w")
f4 = open("fixTest.txt", "w")

f1.write(text1)
f1.close()
f2.write(text2)
f2.close()
f3.write(text3)
f3.close()
f4.write(text4)
f4.close()



