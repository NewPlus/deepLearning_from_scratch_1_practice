class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y #역전파에서 곱셈 노드는 x,y 값을 바꾼다.
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num) # 1
orange_price = mul_orange_layer.forward(orange, orange_num) # 2
all_price = add_apple_orange_layer.forward(apple_price, orange_price) # 3
price = mul_tax_layer.forward(all_price, tax) # 4

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) # 4
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) # 3
dorange, dorange_num = mul_orange_layer.backward(dorange_price) # 2
dapple, dapple_num = mul_apple_layer.backward(dapple_price) # 1

print(price) # 순전파 결과 값 : 소비세가 적용된(1.1) 사과 2개 + 오렌지 3개 값은?
print(dapple_num, dapple, dorange ,dorange_num, dtax) # 역전파 결과 값 : 소비세가 아주 약간 오르면 가격은 얼마나 오르나?