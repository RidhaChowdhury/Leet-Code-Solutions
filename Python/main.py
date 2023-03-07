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

def decodeString(self, s: str) -> str:
        result = ""
        k_codes = []
        reading_number = False
        # parse the string adding
        for index, char in enumerate(s):
            # add characters to the result string if the stack is empty
            if not (char.isdigit() or char == '[' or char == ']'):
                if not k_codes:
                    result += char
                else:
                    k_codes[-1][1] += char

            # when a number is found store the number and start reading the corresponding text
            if char.isdigit():
                # start reading the number
                if not reading_number:
                    k_codes.append([int(char), ""])
                    reading_number = True
                
                else:
                    k_codes[-1][0] *= 10
                    k_codes[-1][0] += int(char)
            
            if char == '[':
                reading_number = False

            # once a closing bracket is found perform the string multiplication and append it to the value below it on the stack
            if char == ']':
                new_parse = k_codes[-1][0]*k_codes[-1][1]
                k_codes.pop()
                # if the stack is empty append it to the result string
                if not k_codes:
                    result += new_parse
                # otherwise append it to the k_code under it in the stack, which it is nested in
                else:
                    k_codes[-1][1] += new_parse


        return result

def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # have a stack of remaining asteroids
        remaining = []

        # traverse asteroids
        for asteroid in asteroids:
            # if the asteroid is + just push it onto the stack, it wont collide with any previous asteroids
            # if the stack is empty also just push onto the stack, theres nothing to collide with
            if not remaining or asteroid > 0:
                remaining.append(asteroid)

            # if the asteroid is (-) loop collision evaluation
            else:
                # keep evaluating while there are right moving asteroids to collide with
                while remaining and remaining[-1] > 0:
                    # check if its an equivalent collision
                    if(remaining[-1] == abs(asteroid)):
                        remaining.pop()
                        break
                    # if the asteroid on top would destory the current asteroid, break
                    if(remaining[-1] > abs(asteroid)):
                        break
                    # otherwise destroy right moving asteroid
                    else:
                        remaining.pop()

                # our asteroid survived all the collisions so push it onto the stack
                else:
                    remaining.append(asteroid)

        # return the remaining asteroids
        return remaining

def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        answer = [0 for i in range(len(temperatures))]
        temperature_history = []
        
        # go through each day
        for day, temperature in enumerate(temperatures):
            # start looking through the stack popping values and appending
            while temperature_history and temperature_history[-1][1] < temperature:
                popped_temperature = temperature_history.pop()
                answer[popped_temperature[0]] = day - popped_temperature[0]
            
            # slap the new temperature on the stack
            temperature_history.append((day, temperature))

        # all the temperatures left over where a warmer day wasnt found
        for day_temp in temperature_history:
            answer[day_temp[0]] = 0
        
        return answer

def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        largest = heights[0]
        # brute force
        for new_height in heights:
            stack.append(new_height)
            smallest = new_largest = new_height
            for index, old_height in enumerate(reversed(stack)):
                smallest = min(smallest, old_height)
                new_largest = max(smallest * (index + 1), new_largest)

            largest = max(new_largest, largest)
        
        return largest

def hasCycle(self, head: Optional[ListNode]) -> bool:
        seen = set()
        while head is not None:
            if head in seen:
                return True
            seen.add(head)
            head = head.next

        return False

        '''# have a slow and fast pointer
        slow = head
        if head is not None:
            fast = head.next
        else:
            return False
        
        # traverse while the slow and fast pointer are still not none
        while (slow is not None) and (fast is not None):
            # if we find that the fast pointer is the same as slow or slows next return true
            if slow == fast or fast.next == slow:
                return True
            # if fast cant double jump without encountering none we know there is no cycle
            if fast.next is None:
                return False

            slow = slow.next
            fast = fast.next.next

        return False'''

def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # have a dummy answer list and a carry over value
        answer = tail = ListNode()
        carry = 0

        # while there are still numbers in both lists continue adding
        while l1 is not None and l2 is not None:
            # add the values of the nodes and carry
            sum = l1.val + l2.val + carry

            # set the value of a new list node on tail to the sum % 10
            tail.next = ListNode(sum % 10)

            # set carry to the sum // 10 to get the second digit of the addition 
            carry = sum // 10

            # advance the list's
            l1, l2, tail = l1.next, l2.next, tail.next


        
        # after what ever is remaining of either list can be attached to the end
        extra_digits = adding = l1 if l2 is None else l2
        while carry != 0 and adding is not None:
            if adding.val == 9:
                adding.val = 0
            else:
                adding.val += carry
                carry = 0
                break

            # if we've reached a final digit
            if adding.next is None:
                adding.next = ListNode(carry)
                carry = 0

            adding = adding.next

        # covers edge case
        if carry != 0 and extra_digits is None:
            tail.next = ListNode(carry)
        else:
            tail.next = extra_digits

        return answer.next

class LRUCache:
    # overall structure
    # use hashmap in conjunction with linked list
    # hashmap for storing key value pairs
    # linked list for use history and to know what to pop when
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {} # key, [value, history_freq]
        self.most_recent = ListNode(-1)
        self.least_recent = self.most_recent
        

    def get(self, key: int) -> int:
        # try to get the value from the map
        value = self.map[key][0] if key in self.map else -1
        
        # if we couldnt find it return -1
        if value == -1:
            return -1

        # when you get something move it to the back of the linked list
        self.most_recent.next = ListNode(key)
        self.most_recent = self.most_recent.next

        # and increase it's frequency
        self.map[key][1] += 1

        return value
        
        

    def put(self, key: int, value: int) -> None:
        # add it to the hashmap
        if key in self.map:
            self.map[key] = [value, self.map[key][1] + 1]
        else:
            self.map[key] = [value, 1]

        # add it to the history
        self.most_recent.next = ListNode(key)
        self.most_recent = self.most_recent.next

        if self.least_recent.value == -1:
            self.least_recent = self.least_recent.next

        print(key, value)
        # if the hashmap exceeds capacity, start trying to clear from the map
        while len(self.map) > self.capacity:
            head_key = self.least_recent.value

            self.map[head_key][1] -= 1

            if self.map[head_key][1] == 0:
                self.map.pop(head_key)
            
            self.least_recent = self.least_recent.next

def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # # handle the single node edge case
        # if head.next is None:
        #     return None

        # utilize lead lag and a dummy pointer
        lead = head
        dummy = lag = ListNode(-1, head)
        traversed = 0

        # while the lead pointer isn't null
        while lead is not None:
            # jump lead forward in the linked list
            lead = lead.next
            traversed += 1

            # if we've traversed enough without lag, bring it forward aswell
            lag = lag.next if traversed > n else lag


        # lag.next should be our n from end node
        lag.next = lag.next.next

        # return the head
        return dummy.next

def reorderList(self, head: Optional[ListNode]) -> None:
        # make a dummy reversed list, while doing so find the length of it
        dummy = ListNode(head.val)
        
        current = head.next
        list_size = 0
        while current:
            new_node = ListNode(current.val, dummy)
            dummy = new_node
            current = current.next
            list_size += 1
        
        # interlace the lists
        current = head
        for i in range(list_size//2):
            print(current.val, current.next.val, dummy.val, dummy.next.val)
            cur_next, dum_next = current.next, dummy.next
            current.next, dummy.next = dummy, current.next
            current, dummy = cur_next, dum_next
        
        if list_size % 2:
            current.next.next = None
        else:
            current.next = None

        return head

def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # have a current pointer and a value
    current = head
    value = -101

    # traverse the linked list while current is not none
    while current is not None:
        # set value to currents value
        value = current.val
        
        # make a similar seeking pointer set to current
        seeking = current

        # while the value of seeking is the same as value keep seeking
        while seeking is not None and seeking.val == value:
            seeking = seeking.next

        # connect current to seeking and set current to seeking
        current.next = seeking
        current = current.next
        
    # return head
    return head

def guessNumber(self, n: int) -> int:
        low, high = 1, n
        while low <= high:
            middle = (low + high) // 2
            direction = guess(middle)
            if direction == -1:
                high = middle - 1
            elif direction == 1:
                low = middle + 1
            else:
                return middle

def mySqrt(self, x: int) -> int:
        low, high = 0, x
        closest = 0
        while low <= high:
            middle = (low + high) // 2
            square = middle * middle
            if square > x: 
                high = middle - 1
            elif square < x:
                low = middle + 1
                closest = max(closest, middle)
            else: 
                return middle
        return closest

def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        middle = (left + right) // 2
        while left<=right:
            if nums[middle] < target:
                left = middle + 1
            elif nums[middle] > target:
                right = middle - 1
            else:
                return middle
            middle = (left + right) // 2
        return -1

def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # Strategy: implement a binary search with a search function to abstract the 2d element of the data
        # dimensions of the array
        m = len(matrix)
        n = len(matrix[0])

        # method for searching the matrix as if it was a 1d list
        def search(index):
            return matrix[index // n][index % n]

        # the binary search
        left, right = 0, m * n - 1

        while left <= right:
            middle = (left+right) // 2
            value = search(middle)
            if value < target:
                left = middle + 1
            elif value > target:
                right = middle - 1
            else:
                return True
        return False

def search(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    in_left = target >= nums[0]
    while left <= right:
        middle = (left + right) // 2
        center_num = nums[middle]
        # if its in the left half and we're looking at the right bring the right to middle
        if in_left and center_num < nums[0]:
            right = middle - 1
            continue

        # if its in the right half and we're looking at the left bring the left to middle
        elif not in_left and center_num >= nums[0]:
            left = middle + 1
            continue

        # otherwise do standard binary search behaviour
        if center_num > target:
            right = middle - 1
        elif center_num < target:
            left = middle + 1
        else:
            return middle

    return -1

def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # find how long it would take for koko to eat 
        def time_to_eat(speed):
            time = 0
            for pile in piles:
                time += pile // speed
                time += 1 if pile % speed != 0 else 0
            return time
        
        slowest, fastest = 1, max(piles)
        minimum = fastest
        while slowest <= fastest:
            # evaluate how long it would take for the middle speed
            middle = (slowest + fastest) // 2
            time = time_to_eat(middle)

            # if koko can eat all the bananas at this middle speed within h chop out the faster speeds
            if time <= h:
                fastest = middle - 1
                minimum = min(middle, minimum)
            # otherwise koko needs to speed up
            else:
                slowest = middle + 1

        return minimum

def findMin(self, nums: List[int]) -> int:
        # left and right pointer
        left, right = 0, len(nums) - 1

        # handle edge case where its rotated n times
        if nums[left] < nums[right]:
            return nums[left]

        # move the right pointer if middle is in the wrapped half otherwise move the left pointer
        while left < right:
            middle = (left + right) // 2
            mid_val = nums[middle]
            # number is in the wrapped half
            if mid_val >= nums[0]:
                left = middle + 1
            # number is in the unwrapped half
            else:
                right = middle

        return nums[left]

def maxDepth(self, root: Optional[TreeNode]) -> int:
    if root == None:
        return 0
    
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []

        if root.left is None and root.right is None:
            return [[root.val]]

        node_stack = [[root]]
        current_depth, current_index = 0, 0
        # traverse the node stack
        while current_depth < len(node_stack) and node_stack[-1]:
            # make a new stack for the next children
            node_stack.append([])
            while current_index < len(node_stack[current_depth]):
                current_node = node_stack[current_depth][current_index]
                node_stack[current_depth][current_index] = current_node.val
                
                # add each node's children to the next substack in the node stack
                if current_node.left is not None:
                    node_stack[current_depth + 1].append(current_node.left)
                if current_node.right is not None:
                    node_stack[current_depth + 1].append(current_node.right)
                
                current_index += 1

            # reset pointers
            current_index = 0
            current_depth += 1

        # remove the 
        node_stack.pop()

def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    # if we find a structural difference
    if (p is None) ^ (q is None):
        return False

    # if the tree terminates here
    elif p is None and q is None:
        return True

    # otherwise compare values
    elif p.val == q.val:
        # if the value is the same continue comparing
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    else:
        # otherwise return false
        return False
    
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:        
    def rightSideRecursive(root, current_level, next_level):
        res = []

        if root is None:
            return res

        # if we're at the next required level, since its preorder VRL we know this is the rightmost next_level node
        if current_level == next_level:
            res.append(root.val)
            next_level += 1


        # do the search down the left leg
        right_results = rightSideRecursive(root.right, current_level + 1, next_level)
        res.extend(right_results) # add its values to the result
        next_level+= len(right_results) # however many we were able to get down the left leg advance the next desired level by that amount

        res.extend(rightSideRecursive(root.left, current_level + 1, next_level)) # similarly for the left leg

        return res
    return rightSideRecursive(root, 1, 1)

def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # going down the right path would not split the 
    if (root.val > p.val) and (root.val > q.val) and root.right:
        return self.lowestCommonAncestor(root.left, p, q)
    # going down the left path
    elif (root.val < p.val) and (root.val < q.val) and root.left:
        return self.lowestCommonAncestor(root.right, p, q)
    # they split
    else:
        return root
    
def goodNodes(self, root: TreeNode) -> int:
    # have a helper method that keeps track of the max node passed while traversing
    def goodNodesHelper(root, max):
        if root is None:
            return 0

        if max > root.val:
            return goodNodesHelper(root.left, max) + goodNodesHelper(root.right, max)
        else:
            return 1 + goodNodesHelper(root.left, root.val) + goodNodesHelper(root.right, root.val)

    return goodNodesHelper(root, -float("inf"))

def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    # handle nodeless edge case
    if not root:
        return TreeNode(val)

    # initialize pointer values, they will be useful in the tree traversal
    last = slot = root

    #keep running down the BST while looking at nodes to discover the parent node
    while slot:
        last = slot
        if slot.val > val:
            slot = slot.left
        else:
            slot = slot.right
        
    # attach the value to the corresponding child of its discovered parent
    if last.val > val:
        last.left = TreeNode(val)
    else:
        last.right = TreeNode(val)

    return root

def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # handle edge case
    if head is None or head.next is None:
        return head

    # a pair swapping utility
    def swap_pair(first):
        if not first:
            return None, None

        if not first.next:
            return first, None
    
        second = first.next
        first.next = second.next
        second.next = first
        return second, first

    # swap the first pair
    new_head, next_pair = swap_pair(head)

    # while there is another pair to swap perform the swap
    while next_pair:
        # make the connection and swap the next pair
        next_pair.next, next_pair = swap_pair(next_pair.next)


    return new_head

def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
  levels = [[root]]
  current_level = 0

  while len(levels[current_level]) > 0:
      # add a stage for the next level
      levels.append([])
      
      for node in levels[current_level]:
          if node.left:
              levels[-1].append(node.left)
          if node.right:
              levels[-1].append(node.right)

      current_level += 1

  return levels[current_level - 1][0].val