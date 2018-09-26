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
// else use intrusive deleter

#define USE_FIXEDPOOL


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

template<class T>
struct IDeallocator
{
    virtual void deallocate(T* p) = 0;
    virtual ~IDeallocator() {}
};

template<class A, class T>
using member_deallocate_pointer = void (A::*)(T*); 

struct RefDeallocator_Tag
{
};

struct CopyDeallocator_Tag
{
};

// Deallocator implements deallocato(T*)
template<class T, class SaveDeallocTag = RefDeallocator_Tag , class Deallocator = IDeallocator<T>, member_deallocate_pointer<Deallocator, T> DelFunc = &Deallocator::deallocate>
struct IDeallocatable
{
    using value_type = T;
    using deallocator_type = Deallocator;
    static constexpr bool is_deallocator_copied = std::is_same<SaveDeallocTag, CopyDeallocator_Tag>::value;
    using saved_deallocator_type = std::conditional_t< is_deallocator_copied, std::optional<deallocator_type>,  deallocator_type*>;

//    static constexpr saved_deallocator_type null_deallocator = std::conditional_t< std::is_same_v<SaveDeallocTag, CopyDeallocator_Tag>, std::nullopt,  nullptr>;

    IDeallocatable(deallocator_type &dealloc) {
        set_deallocator(dealloc);
    }

    IDeallocatable() {}

    IDeallocatable(const IDeallocatable& ) {}
    IDeallocatable& operator=(const IDeallocatable&) {
        return this;
    }

    void set_deallocator( deallocator_type &dealloc) {
        if constexpr( is_deallocator_copied )
            mDeallocator = dealloc;
        else
            mDeallocator = &dealloc;
    }
    void reset_deallocator() {
            mDeallocator = saved_deallocator_type();
    }

    const auto &get_deallocator() const {
        return mDeallocator;
    }

    auto &get_deallocator()  {
        return mDeallocator;
    }

    void deallocate_me() {
        if( auto& d = get_deallocator() ) {
            reset_deallocator();
            static_cast<T*>(this)->~T(); // call destructor
            ((*d).*DelFunc)(static_cast<T*>(this));
        }
    }

    virtual ~IDeallocatable() {
        if( auto& d = get_deallocator() ) {
            reset_deallocator();
            ((*d).*DelFunc)(static_cast<T*>(this));
        }
    }
private:
    saved_deallocator_type mDeallocator = saved_deallocator_type();
};

template <class T, class D>
using IDeleter = IDeallocatable<T, CopyDeallocator_Tag, D, &D::operator()>;


struct Node : IDeallocatable<Node>
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
struct null_delete{
    void operator()(T* ){}
};

template<class T, class Deleter = std::default_delete<T> >
struct FreeList
{
    struct FreeNode
    {
        FreeNode *pNext;

        FreeNode(T *next = nullptr):pNext( reinterpret_cast<FreeNode*>(next)){}
        FreeNode(FreeNode *next = nullptr):pNext(next){}
    };

    FreeList(T *p = nullptr, const Deleter& d ={}): pHead( reinterpret_cast<FreeNode*>(p)),mDel(d) {}

    FreeList(const Deleter& d):mDel(d){}

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

    template<class D = Deleter>
    typename std::enable_if<std::is_same<D, null_delete<T>>::value, void >::type clear() {
        pHead = nullptr;
    }

    template<class D = Deleter>
    typename std::enable_if<!std::is_same<D, null_delete<T>>::value, void >::type clear() {
        while( auto p=pop() ) {
            mDel( p );
        }
        pHead = nullptr;
    }

    FreeNode *pHead = nullptr;
    Deleter mDel;
};



// allocate when constructing BasePool
// ChildT implements: T *do_alloc()
template<class T, class FreeList, class ChildT >
class BasePool : IDeallocator<T>
{
public:
    using child_type = ChildT;
//    using T = typename ChildT::value_type;
//    using FreeList = typename ChildT::freelist_type;


    BasePool( const FreeList& fl): freeList(fl) {}

    FreeList& get_freelist(){
        return freeList;
    }

    T* allocate()
    {
        if( auto p = freeList.pop() )
            return p;
        else {
//            auto pChild = dynamic_cast<child_type*>(this);
//            assert(pChild && "Unable to get child");
//            return pChild->do_alloc();
            return do_alloc();
        }
    }

    virtual T* do_alloc() = 0;

    template<class... Args>
    T* create(Args&&... args) {
        if( auto p = allocate() ) {
            new(p) T(std::forward<Args>(args)...);
            p->set_deallocator(*this);
            return p;
        }
        return nullptr;
    }

    template<class... Args>
    std::shared_ptr<T> create_shared(Args&&... args)
    {
        if( auto p = allocate() ) {
            new (p) T(std::forward<Args>(args)...);
            return {p, [this](T* pObj){ deallocate(pObj);} };
        }
        return nullptr;
    }

    template<class... Args>
    std::unique_ptr<T, DeleteFunc<T>> create_unique(Args&&... args)
    {
        if( auto p = allocate() ) {
            new (p) T(std::forward<Args>(args)...);
            return {p,[this](T* pObj){ deallocate(pObj);} };
        }
        return nullptr;
    }

    void deallocate(T* p) override
    {
        assert(p);
        freeList.push(*p);
    }

    void clear() // reset
    {
        freeList.clear();
    }

protected:
    FreeList freeList;
};

// allocate when custructing FreePool
template<typename T, class Alloc = std::allocator<T> , class FreeList = FreeList<T, null_delete<T>> >
class FixedPool : BasePool<T, FreeList, FixedPool<T, Alloc,FreeList> >
{
public:
    using this_type = FixedPool;
    using parent_type = BasePool<T, FreeList, this_type >;

    using parent_type::create;

    // pre-allocate memory. memory will be deleted when FreePool destructs.
    // T must be IDeallocatable
    FixedPool(size_t nBlocks, const Alloc &a=Alloc(), const FreeList& freeList=FreeList())
        : parent_type(freeList), mAlloc(a), mBlocks(nBlocks)
    {
        if( !nBlocks ) return;
        mCurrent = mMem = mAlloc.allocate(mBlocks);
        assert(mMem);
        if(!mMem) {
            throw std::bad_alloc();
        }
    }

    ~FixedPool() 
    {
        if(mMem) {
            mAlloc.deallocate( mMem, mBlocks );
            mMem = nullptr;
        }
    }

    T* do_alloc()
    {
        if( mCurrent + 1 > mMem + mBlocks ) return nullptr;
        auto res = (mCurrent);
        mCurrent += 1;
        return res;
    }

    void clear() // reset
    {
        parent_type::clear();
        mCurrent = mMem;
    }

protected:
    Alloc mAlloc;
    const size_t mBlocks;
    T *mMem = nullptr;
    T *mCurrent;
};

// allocate when needed
template<typename T, class Alloc = std::allocator<T> , class FreeList = FreeList<T, std::default_delete<T>> >
class PoolAllocator  : BasePool<T, FreeList, PoolAllocator<T, Alloc, FreeList> >
{
public:
    using this_type = PoolAllocator;
    using parent_type = BasePool<T, FreeList, this_type >;

    using parent_type::create;


    // pre-allocate memory. memory will be deleted when FreePool destructs.
    // T must be IDeallocatable
    PoolAllocator( const Alloc &a=Alloc(), const FreeList& freeList=FreeList())
        : parent_type(freeList), mAlloc(a)
    {
    }

    PoolAllocator( size_t n, const Alloc &a=Alloc(), const FreeList& freeList=FreeList())
        : parent_type(freeList), mAlloc(a)
    {
    }

    T* do_alloc()
    {
        mAlloc.allocate(1);
    }

    void clear() // reset
    {
        parent_type::clear();
    }

protected:
    Alloc mAlloc;
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

    T* create()
    {
        if( mCurrent + BlockSize > mMem + mMemSize ) return nullptr;
        auto res = reinterpret_cast<T*> (mCurrent);
        mCurrent += BlockSize;
        return res;
    }

    void deallocate(T* p)
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

#ifdef USE_FIXEDPOOL
typedef FixedPool<Node> NodePool;
#else
typedef PoolAllocator<Node> NodePool;
#endif

//typedef Pool<Node> NodePool;

NodePtr make(int d, NodePool &store)
{
#ifdef ALLOC_SHARED
    auto root = store.create_shared();
#elif defined( ALLOC_UNIQUE )
    auto root = store.create_unique();
#else
    auto root = store.create();
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
