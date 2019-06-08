
import svhn

def main():

    result1 = svhn.test("mytestimage1.jpg")
    print(result1)

    result2 = svhn.test("mytestimage2.png")
    print(result2)


    average_f1_scores = svhn.traintest()
    print(average_f1_scores)


if __name__ == '__main__':
    main()
