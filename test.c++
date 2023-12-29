#include <CL/sycl.hpp>
constexpr int N=999999;
constexpr int B=4;
using namespace sycl;

int main() {
 queue q;
 auto data = malloc_shared<int>(N, q);
  	int j,i;
	double pi;
	double dx;
		
	dx = 1.0/N;
	
	
  
  
 for ( i = 0; i < N; i++) data[i]=i;
  
 auto area = malloc_shared<double>(N,q);
  
 q.parallel_for(nd_range<1>{N, B}, 
             reduction(area, plus<>()), [=](nd_item<1> it,auto& temp)
 {
	int i = it.get_global_id(0);
	double x = data[i]*dx;
	double y = sqrt(1-x*x);
   	temp += y*dx;

     
 }).wait();
  
 pi=4.0*area[0];
  
 std::cout << "pi = " << pi << std::endl;

return 0;
}
