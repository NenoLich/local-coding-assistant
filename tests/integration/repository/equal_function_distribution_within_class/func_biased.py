class TestClass:
    def biased_function(self):
        pass

    def unbiased_function_1(self):
        self.biased_function()

    def unbiased_function_2(self):
        self.biased_function()

    def unbiased_function_3(self):
        self.biased_function()


class Outsider:
    def outsider_function_1(self):
        test = TestClass()
        test.biased_function()

    def outsider_function_2(self):
        test = TestClass()
        test.biased_function()

    def outsider_function_3(self):
        test = TestClass()
        test.biased_function()
