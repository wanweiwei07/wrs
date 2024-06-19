class Car(object):
    def __init__(self, brand, speed):
        self.brand = brand
        self.speed = speed

    def accelerate(self):
        self.speed += 10
        print(f"The car is now going at {self.speed} km/h")

class ElectricCar(Car):
    def __init__(self, brand, speed, battery):
        super().__init__(brand, speed)  # 親クラスの初期化を呼び出す
        self.battery = battery  # 新しい属性を追加

    def charge(self):
        print(f"The battery is now charged to {self.battery} kWh")

if __name__ == '__main__':
    my_electric_car = ElectricCar("Tesla", 70, 100)
    my_electric_car.accelerate()
    my_electric_car.charge()
