/*
 *
 * A simple workspace for working with leetcode problems, they run faster here and can be published to my repo
 */
#include<iostream>
#include<vector>
#include<unordered_map>
#include<algorithm>

using namespace std;

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
#pragma endregion


int main() {

}
