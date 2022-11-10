/*
 *
 * A simple workspace for working with leetcode problems, they run faster here and can be published to my repo
 */
#include<iostream>
#include<vector>
#include<unordered_map>
#include<algorithm>

using namespace std;
#pragma region dataStructures
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
#pragma endregion

#pragma region helperFunctions

template <typename T>
void swap(T* a, T* b) { // Helper function for reverseString
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

#pragma endregion

#pragma region completedSolutions
bool isHappy(int n) { // https://leetcode.com/problems/happy-number/
    int digitSum = 0;
    int num = n;
    unordered_map<int, bool> history;
    while (digitSum != 1) {
        if (history[num]) return false;
        history[num] = true;
        digitSum = 0;
        while (num != 0) {
            int digit = num % 10;
            digitSum += digit * digit;
            num /= 10;    // Chopping processed number off of n
        }
        num = digitSum;
    }
    return true;
}

void isHappyImplementation(int dataSetSize) {
    for(int i = 0; i < dataSetSize; i++) {
        string happy = isHappy(i) ? " is happy!" : " sad";
        cout << i << happy << "\n";
    }
}

void reverseString(vector<char>& s) { //https://leetcode.com/problems/reverse-string/
    for(int i = 0; i < s.size()/2; i++) {
        swap(&s[i], &s[s.size() - i - 1]);
    }
}

void deleteNode(ListNode* node) { // https://leetcode.com/problems/delete-node-in-a-linked-list/
    node->val = node->next->val;
    if(node->next->next == NULL){
        node->next = NULL;
        return;
    }
    return deleteNode(node->next);
}

vector<int> twoSum(vector<int>& nums, int target) { // https://leetcode.com/problems/two-sum/
    unordered_map<int, int> numbers;
    for(int i = 0; i < nums.size(); i++) {
        if(numbers.count(target - nums[i])) return {i, numbers[target-nums[i]]};
        else numbers[nums[i]] = i;
    }
    return {0,0};
}

int finalValueAfterOperations(vector<string>& operations) { // https://leetcode.com/problems/final-value-of-variable-after-performing-operations/
    int x = 0;
    for(string operation : operations) {
        if(operation == "X++" || operation == "++X") x++;
        else x--;
    }
    return x;
}


int removeDuplicates(vector<int>& nums) { // https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    int last = 0;
    for(int i = 1; i < nums.size(); i++) {
        if(nums[last] != nums[i]) {
            last++;
            nums[last] = nums[i];
        }
    }
    return last + 1;
}

int maxDepth(TreeNode* root) { // https://leetcode.com/problems/maximum-depth-of-binary-tree/
    if(root==NULL) return 0;
    return max(maxDepth(root->left), maxDepth(root->right)) + 1;
}

int smallestEvenMultiple(int n) { // https://leetcode.com/problems/smallest-even-multiple/
    if(n % 2 == 0) return n;
    return 2*n;
}

vector<int> inorderTraversal(TreeNode* root) { // https://leetcode.com/problems/binary-tree-inorder-traversal/
    if(root==nullptr) return {};
    if(root->left == nullptr && root->right == nullptr) return {root->val};
    vector<int> inorder;
    if(root->left) {
        vector<int> left = inorderTraversal(root->left);
        if(root->left->val) inorder.insert(inorder.end(), left.begin(), left.end());
    }
    inorder.push_back(root->val);
    if(root->right) {
        vector<int> right = inorderTraversal(root->right);
        if(root->right->val) inorder.insert(inorder.end(), right.begin(), right.end());
    }
    return inorder;
}

vector<int> shuffle(vector<int>& nums, int n) { // https://leetcode.com/problems/shuffle-the-array/
    vector<int> shuffled;
    for(int i = 0; i < n; i++) {
        shuffled.push_back(nums[i]);
        shuffled.push_back(nums[i + n]);
    }
    return shuffled;
}

int numIdenticalPairs(vector<int>& nums) { // https://leetcode.com/problems/number-of-good-pairs/
    int pairs = 0;
    for(int i = 0; i < nums.size(); i++) {
        for(int j = i + 1; j < nums.size(); j++) {
            if(nums[i] == nums[j]) pairs++;
        }
    }
    return pairs;
}

ListNode* reverseListIterative(ListNode* head) { // https://leetcode.com/problems/reverse-linked-list/
    if(head == nullptr || head->next == nullptr) return head;
    ListNode* newHead = head->next;
    ListNode* previous = head;
    previous->next = nullptr;
    while(newHead != nullptr) {
        ListNode* buffer = newHead->next;
        newHead->next = previous;
        previous = newHead;
        newHead = buffer;
    }
    return previous;
}

int majorityElement(vector<int>& nums) { // https://leetcode.com/problems/majority-element/
    /*unordered_map<int, int> frequency;
    int majority = nums.size() / 2;
    for(int num : nums) {
        frequency[num]++;
        if(frequency[num] > majority) return num;
    }

    return -1;*/

    int frequency = 0, major = nums[0];
    for(int n : nums) {
        if(n == major) frequency++;
        else {
            frequency--;
            if(frequency < 0) {
                frequency = 1;
                major = n;
            }
        }
    }
    return major;
}

void moveZeroes(vector<int>& nums) { // https://leetcode.com/problems/move-zeroes/
    int back = 0;
    for(int i = 0; i< nums.size(); i++) {
        if(nums[i] != 0) {
            swap(nums[back], nums[i]);
            back++;
        }

    }
}

int numJewelsInStones(string jewels, string stones) { // https://leetcode.com/problems/jewels-and-stones/
    unordered_map<char, bool> jewelMap;

    for(char jewel : jewels) {
        jewelMap[jewel] = true;
    }

    int sum = 0;
    for(char stone : stones) {
        if(jewelMap[stone]) sum++;
    }

    return sum;
}

class ParkingSystem { // https://leetcode.com/problems/design-parking-system/
public:
    int spots[3] = {};
    ParkingSystem(int big, int medium, int small) {
        spots[0] = big;
        spots[1] = medium;
        spots[2] = small;
    }

    bool addCar(int carType) {
        carType--;
        if(spots[carType] == 0) return false;
        spots[carType]--;
        return true;
    }
};

int maximum69Number(int num) { // https://leetcode.com/problems/maximum-69-number/
    int* digits = new int[6];
    int currentDigit = 0;
    int baseNum = num;
    int maxNumber = num;
    while(num != 0) {
        digits[currentDigit] = num % 10;
        num /= 10;
        currentDigit++;
    }

    for(int i = 0; i < currentDigit; i++) {
        if(digits[i] == 6) {
            int number = 0;
            bool swapUsed = false;
            for(int j = currentDigit - 1; j >= 0; j--) {
                number *= 10;
                if(j!=i) number += digits[j];
                else number += 9;
            }
            if(number > maxNumber) maxNumber = number;
        }
    }
    return maxNumber;
}

int subtractProductAndSum(int n) { // https://leetcode.com/problems/subtract-the-product-and-sum-of-digits-of-an-integer/
    int product = n % 10;
    int sum = n % 10;
    n /= 10;
    while(n!=0) {
        product *= n % 10;
        sum += n % 10;
        n /= 10;
    }
    return product - sum;
}

#pragma endregion