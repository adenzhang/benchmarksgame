/* The Computer Language Benchmarks Game
 * https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
 *
 * contributed by Jon Harrop
 * modified by Alex Mizrahi
 * modified by Andreas Schäfer
 * very minor omp tweak by The Anh Tran
 * modified to use apr_pools by Dave Compton
 *  *reset*
 */

#include <iostream>
#include <cassert>
#include <exception>
#include <chrono>
#include <memory>
#include <functional>

//#include <apr_pools.h>

//#define ALLOC_SHARED
//#define ALLOC_UNIQUE
// todo intrutrsive shared ptr
//
const size_t    LINE_SIZE = 64;

/*
class Apr
{
public:
    Apr() 
    {
        apr_initialize();
    }

    ~Apr() 
    {
        apr_terminate();
    }
};
*/

template<class T>
using DeleteFunc = std::function<void(T*)>;

struct Node 
{
#ifdef ALLOC_SHARED
    using Ptr = std::shared_ptr<Node> ;
#elif defined( ALLOC_UNIQUE )
    using Ptr = std::unique_ptr<Node, DeleteFunc<Node>> ;
#else
    using Ptr = Node*;
#endif

    Ptr l = nullptr, r = nullptr;
    
    int check() const 
    {
        if (l)
            return l->check() + 1 + r->check();
        else return 1;
    }
};

using NodePtr =  typename Node::Ptr;

template<class T>
struct FreeList
{
    struct FreeNode
    {
        FreeNode *pNext;

        FreeNode(T *next = nullptr):pNext( reinterpret_cast<FreeNode*>(next)){}
        FreeNode(FreeNode *next = nullptr):pNext(next){}
    };

    FreeList(T *p = nullptr): pHead( reinterpret_cast<FreeNode*>(p)){}

    void push(T &p){
        new (reinterpret_cast<FreeNode*>(&p)) FreeNode(pHead);
        pHead = reinterpret_cast<FreeNode*>(&p);
    }

    T* pop() {
        T* res = reinterpret_cast<T*>(pHead);
        if( pHead ) 
            pHead = pHead->pNext;
        return res;
    }
    bool empty() const {
        return !pHead;
    }
    void clear() {
        pHead = nullptr;
    }

    FreeNode *pHead = nullptr;
};

template<typename T, size_t BlockSize = sizeof(T), class Alloc = std::allocator<T> , class FreeList = FreeList<T> >
class FreePool
{
public:
    // if nBlocks > 0, pre-allocate memory
    // if nBlocks == 0, dynamically allocate memory
    FreePool(size_t nBlocks=0, const Alloc &a=Alloc(), const FreeList& freeList=FreeList())
        : mAlloc(a), freeList(freeList), mMemSize(BlockSize * nBlocks)
    {
        if( !mMemSize ) return;
        mCurrent = mMem = new char[mMemSize];
        assert(mMem);
        if(!mMem) {
            throw std::bad_alloc();
        }
        clear();
    }

    ~FreePool() 
    {
        if( mMem )
        delete [] mMem;
    }

    T* alloc()
    {
        if( auto p = freeList.pop() )
            return p;
        else
            return alloc0();    
    }

    T* alloc0()
    {
        if( !mMemSize ) {
            return mAlloc.allocate(1);
        }
        if( mCurrent + BlockSize > mMem + mMemSize ) return nullptr;
        auto res = reinterpret_cast<T*> (mCurrent);
        mCurrent += BlockSize;
        return res;
    }

    template<class... Args>
    std::shared_ptr<T> allocate_shared(Args&&... args)
    {
        if( auto p = alloc() ) {
            new (p) T(std::forward<Args>(args)...);
            return {p, [this](T* pObj){ deallocate(pObj);} };
        }
        return nullptr;
    }

    template<class... Args>
    std::unique_ptr<T, DeleteFunc<T>> allocate_unique(Args&&... args)
    {
        if( auto p = alloc() ) {
            new (p) T(std::forward<Args>(args)...);
            return {p,[this](T* pObj){ deallocate(pObj);} };
        }
        return nullptr;
    }

    void deallocate(T* p)
    {
        assert(p);
        freeList.push(*p);
    }

    void clear() // reset
    {
        freeList.clear();
        mCurrent = mMem;
    }

protected:
    Alloc mAlloc;
    const size_t mMemSize;
    char *mMem = nullptr;
    char *mCurrent;
    FreeList freeList;
};

template<typename T, size_t BlockSize = sizeof(T)>
class Pool
{
public:
    Pool(size_t nBlocks) 
    {
        mMemSize = BlockSize * nBlocks;
        mMem = new char[mMemSize];
        assert(mMem);
        if(!mMem) {
            throw std::bad_alloc();
        }
        mCurrent = mMem;
    }

    ~Pool() 
    {
        delete [] mMem;
        //apr_pool_destroy(pool);
    }

    T* alloc()
    {
        if( mCurrent + BlockSize > mMem + mMemSize ) return nullptr;
        auto res = reinterpret_cast<T*> (mCurrent);
        mCurrent += BlockSize;
        return res;
    }

    void dealloc(T* p)
    {
    }

    void clear()
    {
        mCurrent = mMem;
        //apr_pool_clear(pool);
    }

protected:
    size_t mMemSize;
    char *mMem;
    char *mCurrent;
};

typedef FreePool<Node> NodePool;
//typedef Pool<Node> NodePool;

NodePtr make(int d, NodePool &store)
{
#ifdef ALLOC_SHARED
    auto root = store.allocate_shared();
#elif defined( ALLOC_UNIQUE )
    auto root = store.allocate_unique();
#else
    auto root = store.alloc();
#endif
    assert(root);

    if(d>0){
        root->l=make(d-1, store);
        root->r=make(d-1, store);
    }else{
//        root->l=root->r=0;
    }

    return std::move(root);
}

#define CalcNodes(nodes)  ((1<< (nodes)+1) -1)


int main(int argc, char *argv[]) 
{
    auto tsStart = std::chrono::steady_clock::now();
//    Apr apr;
    int min_depth = 4;
    int max_depth = std::max(min_depth,
                             (argc == 2 ? atoi(argv[1]) : 10));
    int stretch_depth = max_depth+1;

//    assert( stretch_depth < 0);
    assert( stretch_depth < 32);
    const size_t MaxNodes = CalcNodes(stretch_depth) ; // nodes: 2**(depth+1)-1
//    std::cout << "allocated :" << MemSize << " nodes:" << MaxNodes << " blocksize:" << sizeof(Node) << " stretch_depth:" << stretch_depth << std::endl;
    // Alloc then dealloc stretchdepth tree
    {
        NodePool store(MaxNodes);
        auto c = make(stretch_depth, store);
        std::cout << "stretch tree of depth " << stretch_depth << "\t "
                  << "check: " << c->check() << std::endl;
    }

    NodePool long_lived_store(MaxNodes);
    auto long_lived_tree = make(max_depth, long_lived_store);
    std::cout << "starting..." << std::endl;

    // buffer to store output of each thread
    char *outputstr = (char*)malloc(LINE_SIZE * (max_depth +1) * sizeof(char));

    static FreeList<Node> *s_freeList;
    #pragma omp threadprivate(s_freeList)
    #pragma omp parallel 
    {
        s_freeList = new FreeList<Node>;
    }
    
    #pragma omp parallel for 
    for (int d = min_depth; d <= max_depth; d += 2) 
    {
        auto & freeList= *s_freeList;
        int iterations = 1 << (max_depth - d + min_depth);
        int c = 0;

        // Create a memory pool for this thread to use.
        NodePool store(CalcNodes(d));
//        NodePool store(0, {}, freeList);

        for (int i = 1; i <= iterations; ++i) 
        {
            {
                auto a = make(d, store);
                c += a->check();
            }
            store.clear();
        }

        // each thread write to separate location
        sprintf(outputstr + LINE_SIZE * d, "%d\t trees of depth %d\t check: %d\n", iterations, d, c);
    }

    // print all results
    for (int d = min_depth; d <= max_depth; d += 2) 
        printf("%s", outputstr + (d * LINE_SIZE) );
    free(outputstr);

    std::cout << "long lived tree of depth " << max_depth << "\t "
              << "check: " << (long_lived_tree->check()) << "\n";

    auto tsStop = std::chrono::steady_clock::now();
    std::cout << "- cpp time(ms): " << (tsStop - tsStart).count()/1000/1000 << std::endl;
    return 0;
}
