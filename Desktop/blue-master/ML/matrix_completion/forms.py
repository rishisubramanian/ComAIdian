import numpy as np

class Field:
    """
    Simple class which represents a space for a user to put input.
    """
    def __init__(self, name, prompt=None):
        self.name = name
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = "Please enter an appropriate value for {}: ".format(
                self.name)

    def read_input(self):
        value = input(self.prompt)
        while not(self.is_valid(value)):
            print("That was an inappropriate value.")
            value = input(self.prompt)
        return self.parse(value)

    def parse(self, value):
        return value


class IntegerField(Field):
    """
    This class represents a field which is to be filled in by an integer
    within certain specified bounds.
    """
    def __init__(self, name, lower=None, upper=None):
        """
        Args:
            name (str): The name of the field
            lower (int): The lower bound on the field, None if no lower bound
            upper (int): The upper bound on the field, None if no upper bound
        """
        self.lower = lower if lower is not None else -np.inf
        self.upper = upper if upper is not None else np.inf
        prompt = "Please enter an integer for {} between {} and {}: ".format(
            name, self.lower, self.upper)
        Field.__init__(self, name, prompt)

    def is_valid(self, value):
        """
        Checks if a number is within the bounds for the field.
        
        Args:
            value (str): The string containing the number
        Returns:
            bool: True if the number is within the bounds
        """
        return self.lower <= int(value) <= self.upper

    def parse(self, value):
        """
        This just converts the user input into an integer.

        Args:
            value (str): The string containing the number
        Returns:
            int: The corresponding integer value
        """
        return int(value)


class CategoricalField(Field):
    """
    This represents a field which is to be selected from a collection of
    categories. It will output a list of each of the fields and the user will
    input a number corresponding to the field they desire.
    """
    def __init__(self, name, categories):
        """
        Args:
            name (str): The name of the field
            categories (list[str]): A list of the different categories
        """
        self.categories = categories
        prompt = "Categories:\n"
        for i, category in enumerate(categories):
            prompt += "{}. {}\n".format(i, category)
        prompt += "Please enter the number corresponding to the correct" + \
                  "category for {}: ".format(name)
        Field.__init__(self, name, prompt)

    def is_valid(self, value):
        """
        This will just check if the number inputted actually corresponds to a
        category.

        Args:
            value (str): The number of the category
        Returns:
            bool: True if the number actually corresponds to a category
        """
        number = int(value)
        return 0 <= number < len(self.categories)

    def parse(self, value):
        """
        This will find the correct category based on the index passed by the
        user.
        """
        return self.categories[int(value)]
