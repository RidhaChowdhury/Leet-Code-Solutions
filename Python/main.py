from collections import *

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

class MinStack:

    def __init__(self):
        # have a list that will be treated like a stack
        self.stack = []

        # have a min stack
        self.min_stack = []


    def push(self, val: int) -> None:
        # throw it on top of the stack
        self.stack.append(val)

        # see if min stack needs to be updated, if min_stack is empty have it always be min
        if not self.min_stack or self.min_stack[-1] >= val:
            self.min_stack.append(val)

    def pop(self) -> None:
        # pop the top value, check if value is at top of min_stack, pop both if thats the case
        popped = self.stack.pop()
        
        # see if the min stack needs to be popped as well
        if self.min_stack[-1] == popped:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        # return the top of the min stack
        return self.min_stack[-1]


def reverseList(self, head):
        if not head or not head.next:
            return head
        
        # have a reference to the last node, the current one, and a later
        prev = head
        current = head.next
        prev.next = None
        while current:
            # store currents next in later
            later = current.next
            
            # set current's next to previous
            current.next = prev

            # advance previous to current and current to later
            prev = current
            current = later

        return prev

        '''if not head: return

        # hold on to front node
        true_head = head
        value_stack = []

        current = head
        # push node values onto stack
        while current:
            value_stack.append(current.val)
            current = current.next

        # go to first node and set its value to pop, move onto next, repeat
        current = true_head
        while current:
            current.val = value_stack.pop()
            current = current.next

        return true_head'''
        
        '''if not head.next:
            return head
        self.reverseList(head.next).next = head
        return head'''
        
        
        '''# escape statement in the case where an empty head is given
        if not head: return None
        
        # while there is a current node, remove it and push on to a stack making next node the target\
        node_stack = []
        current = head
        
        while current.next:
            node_stack.append(current)
            current = current.next

        # pop through the stack setting next to the next popped value
        new_head = current
        while node_stack:
            print(current.val)
            current.next = node_stack.pop()
            current = current.next

        # set the old heads next to none
        current.next = None

        return new_head'''

def sockMerchant(n, ar):
    total_pairs = 0
    for socks in Counter(ar):
        total_pairs += socks // 2
    return total_pairs
    
    '''# have a hash map that stores how many of each type of sock we have
    sock_drawer = {}
    total_pairs = 0
    
    # insert each sock into the hashmap
    for sock in ar:
        # if we've already seen another one of these socks
        if sock_drawer.get(sock, False) == True:
            # mark down another pair and reset that sock space in the 'drawer'
            total_pairs += 1
            sock_drawer[sock] = False
        else: sock_drawer[sock] = True
        

    return total_pairs'''

def miniMaxSum(arr):
    total = sum(arr)
    minimum = maximum = arr[0]
    
    for num in arr:
        maximum = max(num, maximum)
        minimum = min(num, minimum)
        
    print(str(total - maximum) + " " + str(total - minimum))

def timeConversion(s):
    hour = s[:2]
    am_or_pm = s[-2:]
    min_sec = s[2:-2]
    
    # if the hour is 12 handle the edge case according to AM/PM
    if(hour == "12"):
        return "00" + min_sec if am_or_pm == "AM" else hour + min_sec
        
    # if its in the am return
    if am_or_pm == "AM":
        return hour + min_sec

    # if its in the pm return it with hour + 12
    else:
        return str(int(hour) + 12) + min_sec

def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        # get the frequency of each number via a dictionary
        frequency = {}
        for num in nums:
            frequency[num] = 1 + frequency.get(num, 0)

        # sort the listified dictionary by its values
        sorted_frequencies = list(frequency.items())
        sorted_frequencies.sort(key = lambda x: x[1]) # O(nlogn) => Tim sort, variant of mergesort

        # pop the last k values out into an answer array
        k_most_frequent = []
        for i in range(k):
            k_most_frequent.append(sorted_frequencies.pop()[0])

        return k_most_frequent

def productExceptSelf(self, nums: List[int]) -> List[int]:
        # make two passes over the nums list, once from the front and once from the back
        # while passing store the "running product," product of all values up to the point
        running_product = 1
        output = [1]*len(nums)
        for i in range(len(nums)):
            output[i] *= running_product
            running_product *= nums[i]

        # reset running product and make the same pass from the back of the list
        running_product = 1
        for i in range(len(nums) - 1, -1, -1):
            output[i] *= running_product
            running_product *= nums[i]

        return output

def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # have a pointer from the back
        for big in range(len(numbers) - 1, 0, -1):
            # see if the smallest number wouldnt bring you into range and skip if so
            if numbers[big] + numbers[0] > target:
                continue
            # advance a pointer from the front seeing if any number adds to the target
            for small in range(0, big):
                # dont look at combinations once they start overshooting target
                if numbers[big] + numbers[small] > target:
                    break
                # if we found the combination return it
                elif numbers[big] + numbers[small] == target:
                    return [small + 1, big + 1]

            
        return [0,0]

def isPalindrome(self, s: str) -> bool:
        # check if a character is alpha numeric by seeing where it falls on the ascii table
        def alpha_num(char):
            return (    
                ord(char) >= ord('A') and ord(char) <= ord('Z') # uppercase letter
                or ord(char) >= ord('a') and ord(char) <= ord('z') # lowercase letter
                or ord(char) <= ord('9') and ord(char) >= ord('0') # number letter
            )             

        # initialize a front and back pointer 
        front, back = 0, len(s) - 1

        # push the pointers together until they meet or pass
        while(front < back):
            # advance the pointers to the next alpha num characte
            while front < back and not alpha_num(s[front]):
                front += 1
                
            while front < back and not alpha_num(s[back]):
                back -= 1
                

            # check if the pointers have the same value, take lower/upper into account
            if s[front].lower() != s[back].lower():
                return False

            # push the pointers to the center
            front += 1
            back -= 1
            
        return True

def minimumDifference(self, nums: List[int], k: int) -> int:
        # handle the edge case where k is 1
        if k == 1:
            return 0
        
        # sort the list
        nums.sort()
        
        # point at two vau
        start_of_range = 0
        minimum = nums[k - 1] - nums[0]
        while start_of_range + k <= len(nums):
            difference = nums[start_of_range + k - 1] - nums[start_of_range]
            start_of_range += 1
            minimum = min(minimum, difference)
        
        return minimum

def moveZeroes(self, nums: List[int]) -> None:
        # handle edge case where there are no 0's
        if 0 not in nums:
            return nums

        # have a tail and nose pointer
        tail, nose = nums.index(0), nums.index(0) + 1

        # loop until the nose reaches the end of the list, because then we've considered every number
        while nose < len(nums):
            # if the number at the nose is not 0 preform a swap
            if nums[nose] != 0:
                nums[tail], nums[nose] = nums[nose], nums[tail]
                # advance the tail to maintain the potential 0 window between nose and tail
                tail += 1

            # always push the nose forward    
            nose += 1
        
        return nums

def reverseString(self, s: List[str]) -> None:
        # have a pointer for the left side of the string and right side
        l, r = 0, len(s) - 1

        # approach the center with both pointers, swapping the characters along the way
        while l < r:
            # preforming the swap
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1

        return s

def maxProfit(self, prices: List[int]) -> int:
        # have a pointer to the cheapest value, this is the left of our window
        # have a pointer which moves quick and evaluates if a new max sell has been found
        cheapest = check = 0
        
        # have a variable which stores the max sell value
        best_profit = 0

        # scan through each day
        for day, price in enumerate(prices):
            # if the price we're looking at is lower than our minimum make that our cheapest value
            if price < prices[cheapest]:
                cheapest = day

            else:
                # compare the selling here to our max
                sell = price - prices[cheapest]
                # update our best_profit
                best_profit = max(sell, best_profit)
        
        # return best profit
        return best_profit

def lengthOfLongestSubstring(self, s: str) -> int:
        # handle edge cases
        if len(s) < 2:
            return len(s)
        # have a back and a front pointer
        back = front = 0
        longest = 0

        # a hashset of characters currently in the window
        characters = set(s[0])

        # loop until the front pointer reaches the end of the string
        while front < len(s) - 1:
            # push the front of the window forward
            front += 1

            # shift the back of the window up until the front character is no longer in the set
            while s[front] in characters:
                characters.remove(s[back])
                back += 1
            
						# add the new character to the set
            characters.add(s[front])

            # otherwise compare if the new window is the longest
            longest = max(longest, front - back + 1)

        # return the longest value
        return longest

def calPoints(self, operations: List[str]) -> int:
        # have a running sum and a stack of scores
        total = 0
        scores = []

        # a helper function that adds new scores to the stack
        def add(newScore: int):
            scores.append(newScore)
            nonlocal total
            total += newScore

        # a function that checks if the string is a number
        def check_int(s):
            if s[0] in ('-', '+'):
                return s[1:].isdigit()
            return s.isdigit()

        # loop through the operations
        for operation in operations:   
            # if the operation is an integer just call add
            if check_int(operation):
                add(int(operation))

            # if the operation is + perform add with the top two numbers
            elif operation == "+":
                add(scores[-1] + scores[-2])

            # if the operation is D perform add with double the top number
            elif operation == "D":
                add(scores[-1] * 2)
            # if the operation is C remove the top number from the stack and deduct it from the total
            elif operation == "C":
                total -= scores.pop()    
                
        return total

class StockSpanner:

    def __init__(self):
        # a list of price, span tuples
        self.prices = []         

    def next(self, price: int) -> int:
        # if this is the first value just pop em on
        if(not self.prices):
            self.prices.append((price, 1))
            return 1

        # have a variable to keep track of what the span will be
        current_span = 1

        # while the value span index's back isn't larger than price add its span to current span
        while self.prices[-current_span][0] <= price:
            # if the next span would point us out of the list this spans the whole list
            if current_span + self.prices[-current_span][1] > len(self.prices):
                # push the price onto the stack with a span of the entire length and return that as span aswell
                current_span = len(self.prices) + 1
                self.prices.append((price, current_span))
                return current_span
            
            # as long as the next current span' span would point us to somewhere still in the list keep searching
            current_span += self.prices[-current_span][1]

            
        # push the price and span onto the stack and return the span
        self.prices.append((price, current_span))
        return current_span
