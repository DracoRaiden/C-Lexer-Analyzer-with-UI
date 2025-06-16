#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<int> numbers = {1, 2, 3, 4, 5};
    int sum = 0;

    for (int i = 0; i < numbers.size(); i++) {
        sum += numbers[i];
    }

    double average = sum / static_cast<double>(numbers.size());

    cout << "The sum of the numbers is: " << sum << endl;
    cout << "The average of the numbers is: " << average << endl;

    if (average > 3.0) {
        cout << "Average is greater than 3.0" << endl;
    } else {
        cout << "Average is less than or equal to 3.0" << endl;
    }

    return 0;
}
