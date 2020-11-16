#include "kmedian.cuh"

template <typename T>
void printHostArray(host_vector<host_vector<T>> host) {
	for (auto i : host) {
		for (auto j : i) {
			cout << j << "\t";
		}
		cout << endl;
	}
}

template <typename T>
host_vector<host_vector<T>> populateHostArray(int num_points = 100) {

	host_vector<host_vector<T>> host;
	srand((unsigned)time(0));

	for (int i = 0; i < num_points; ++i) {
		host_vector<int> point;
		point.push_back(rand());
		point.push_back(rand());
		host.push_back(point);
	}

	return host;
}

int main() {

	auto host = populateHostArray<int>(200);
	KMedians<int> kmedians(host);
	kmedians.fit(10, 5);
	printHostArray<int>(host);

	return 0;
}