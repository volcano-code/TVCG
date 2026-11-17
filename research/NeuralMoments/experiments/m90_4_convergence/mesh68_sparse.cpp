// Reuse the frozen source directly; never edit its numerical predicates.
#include "/data/xvxinyao/Spin_M68_Trellis_Data_Kit/src/mesh68_native.cpp"
extern "C" int m904_outer(const double* v,const int64_t* f,int nf,
 const double* origin,double h,int n,double dx,double dy,
 int64_t* out,int64_t capacity,int64_t* stats){
 try{
  Tree tree(v,f,nf);std::vector<double> hits;hits.reserve(128);
  int64_t used=0;stats[0]=stats[1]=stats[2]=0;
  for(int i=0;i<n;i++)for(int j=0;j<n;j++){
   double x=origin[0]+(i+.5)*h,y=origin[1]+(j+.5)*h;
   hits.clear();ray(tree,0,x+dx*h,y+dy*h,hits);
   std::sort(hits.begin(),hits.end());
   hits.erase(std::unique(hits.begin(),hits.end(),[&](double a,double b){return std::abs(a-b)<1e-9*h;}),hits.end());
   if(hits.size()%2){stats[2]++;continue;}
   for(size_t k=0;k<hits.size();k+=2){
    int lo=std::max(0,int(std::ceil((hits[k]-origin[2])/h-.5)));
    int hi=std::min(n-1,int(std::floor((hits[k+1]-origin[2])/h-.5)))+1;
    if(lo>=hi)continue;
    stats[0]+=hi-lo;
    // Canonical set encoding of the native occupied cells, including subcell gaps.
    if(used && out[4*(used-1)]==i && out[4*(used-1)+1]==j && out[4*(used-1)+3]>=lo)
     out[4*(used-1)+3]=std::max<int64_t>(out[4*(used-1)+3],hi);
    else{
     if(used>=capacity)return -2;
     out[4*used]=i;out[4*used+1]=j;out[4*used+2]=lo;out[4*used+3]=hi;used++;
    }
   }
  }
  stats[1]=used;return stats[2]?1:0;
 }catch(...){return -1;}
}
