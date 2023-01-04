def makeGood(self, s: str) -> str: #https://leetcode.com/problems/make-the-string-great/
        characters = []
        for character in s:
            if characters and characters[-1] == character.swapcase():
                characters.pop()
            else: 
                characters.append(character)
            
        return "".join(characters) 

def pivotIndex(self, nums) -> int:
        left, right = 0, sum(nums)
        for index, value in enumerate(nums):
            left += value
            if left == right:
                return index
            right -= value
        return -1
        
def defangIPaddr(self, address: str) -> str:                
    return address.replace(".", "[.]")

def singleNumber(self, nums: List[int]) -> int:
    result = nums[0]
    for num in nums[1:]:
        result ^= num
    return result

def containsDuplicate(self, nums: List[int]) -> bool:
    # create a hashset
    seen = set()
    # loop through each value
    for num in nums:
        # if the value is in the set then this list contains duplicate
        if num in seen:
            return True
        # otherwise add the value to the set and keep moving
        seen.add(num)
    # if we could traverse the whole list without finding a value already in our set, there are no duplicates
    return False

def isValid(self, s: str) -> bool:
        # dictionary for matching openning brackets
        matching_brackets = {
            "}":"{",
            ")":"(",
            "]":"["
        }

        # make a bracket stack
        stack = []

        # traverse brackets
        for bracket in s:
            # if the bracket is openning put it onto the stack
            if bracket in matching_brackets.values():
                stack.append(bracket)
            # if its closing see if the top of the stack is the corresponding type
            elif len(stack) == 0 or stack.pop() != matching_brackets[bracket]:
                return False

        # if the stack is empty at the end we have no openning brackets left and pass
        return not len(stack)