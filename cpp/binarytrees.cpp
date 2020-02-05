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
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cassert>

const size_t    LINE_SIZE = 64;


template<class T, class AllocT = std::allocator<T>>
class PooledAllocator
{
    struct Node {
        char val[sizeof(T)];
        T *getData() {
            return reinterpret_cast<T*>(val);
        }
        Node *& next(){
            return reinterpret_cast<Node *&>(val);
        }
        static Node *From( T* p) {
            return reinterpret_cast<Node *>(p);
        }
    };
//    static_assert( sizeof(Node) <= sizeof(T), "Too small T");
public:
    struct Deleter{
        PooledAllocator *alloc = nullptr;

        Deleter(PooledAllocator *a= nullptr):alloc(a){}
        void operator()( T* p ) const {
            alloc->deallocate(p);
        }
    };

    using Alloc = typename std::allocator_traits<AllocT>::template rebind_alloc<Node>;
    PooledAllocator(size_t nReserve = 0, const Alloc &alloc = Alloc() ): m_alloc(alloc)
    {
        for(size_t i=0; i< nReserve; ++i) {
            if( auto p = m_alloc.allocate(1) ) {
                deallocate( p->getData() );
            }else
                break;
        }
    }
    T* allocate()
    {
        if( m_freeList ) {
            auto pFirst = m_freeList;
            m_freeList = pFirst->next();
            return pFirst->getData();
        }
        return m_alloc.allocate(1)->getData();
    }

    template<class...Args>
    std::unique_ptr<T, Deleter> create_unique(Args&&... args) {
        auto p = allocate();
        new (p) T( std::forward<Args>(args)... );
        return std::unique_ptr<T, Deleter>( p, Deleter(this) );
    }

    void deallocate( T* p, size_t n = 1) {
        auto pN = Node::From(p);
        pN->next() = m_freeList;
        m_freeList = pN;
        
    }
protected:
    Alloc m_alloc;
    Node *m_freeList = nullptr;
};

struct Node 
{
    using Deleter =  typename PooledAllocator<Node>::Deleter ;
    using Ptr = std::unique_ptr<Node, Deleter > ;

    Ptr l, r;

    Node() = default;
    
    int check() const 
    {
        if (l)
            return l->check() + 1 + r->check();
        else return 1;
    }
};
struct Node0
{
    Node0 *l = nullptr, *r= nullptr;
    
    int check() const 
    {
        if (l)
            return l->check() + 1 + r->check();
        else return 1;
    }
};

using NodePtr = typename Node::Ptr;

class NodePool
{
public:
    NodePool() 
    {
//        apr_pool_create_unmanaged(&pool);
    }

    ~NodePool() 
    {
//        apr_pool_destroy(pool);
    }

    Node0* allocate()
    {
        return static_cast<Node0*>( malloc(sizeof(Node)) );
//        return (Node *)apr_palloc(pool, sizeof(Node));
    }

    void clear()
    {
//        apr_pool_clear(pool);
    }

private:
//    apr_pool_t* pool;
};

Node0* make(int d, NodePool &store)
{
    Node0* root = store.allocate();

    if(d>0){
        root->l=make(d-1, store);
        root->r=make(d-1, store);
    }

    return root;
}

NodePtr make(int d, PooledAllocator<Node> &store)
{
//    Node* root = store.allocate();
    auto root = store.create_unique();

    if(d>0){
        root->l=make(d-1, store);
        root->r=make(d-1, store);
    }

    return root;
}


template<class T, class AllocT = std::allocator<T> >
class SPMCQueue
{
    struct Node{
        std::atomic<bool> flag; // true for having value
        T val;
    };

public:
    using Alloc = typename std::allocator_traits<AllocT>::template rebind_alloc<Node>;
    
    SPMCQueue(size_t nCap, const Alloc& alloc= Alloc() ): m_alloc(alloc), m_data( m_alloc.allocate(nCap+1) ), m_bufsize(nCap+1)
    {
        for(size_t i=0; i< m_bufsize; ++i)
            m_data[i].flag.store( false );
    }
    ~SPMCQueue() {
        if( m_data ) {
            clear();
            m_alloc.deallocate( m_data, m_bufsize );
            m_data = nullptr;
        }
    }

    /// @return false when full. Note: the front element mayb be in process of dequeue.
    bool enqueue(const T& val) {
        auto iEnd =  m_end.load(std::memory_order_acquire) % m_bufsize;
        auto iNext = ( iEnd+1 ) % m_bufsize;
        if( m_data[ iNext ].flag.load(std::memory_order_acquire) ) // full
            return false;
        assert( m_data[iEnd].flag.load(std::memory_order_acquire) == false ); 
        m_data[iEnd].val = val;
        m_data[iEnd].flag.store(true, std::memory_order_release);
        assert( !oldVal );
        m_end.fetch_add( 1 );
        return true;
    }

    bool dequeue(T *val) {
        size_t ibegin = m_begin.load(std::memory_order_acquire);
        int n = 0;
        do{
            if( ibegin == m_end.load(std::memory_order_acquire) ) // emtpy
                return false;
        }while( !m_begin.compare_exchange_weak( ibegin, ibegin+1 ) );  // std::memory_order_acquire

        // alreay acquired the ownership.
        ibegin %= m_bufsize;
        auto& data = m_data[ibegin];
        assert( data.flag.load(std::memory_order_acquire) );

        if( val ) new (val) T ( std::move( data.val ) );
        data.val.~T();
        data.flag.store(false, std::memory_order_release);
        return true;
    }

    size_t size() const {
        return m_end.load() - m_begin.load();
    }

    bool empty() const {
        return m_begin.load() == m_end.load();
    }

    /// weak full. dequeue may be in process.
    bool full() const {
        return m_begin.load() + (m_bufsize - 1 )== m_end.load();
    }

    void stop() {
        m_stopping = true;
    }
    
    bool stopping() const {
        return m_stopping;
    }

    void clear() {
        while( dequeue(nullptr) )
            ;
    }

protected:
    Alloc m_alloc;
    Node *m_data = nullptr;
    size_t m_bufsize = 0;
    std::atomic<size_t> m_begin {0}, m_end {0};
    std::atomic<bool> m_stopping{false};
};

template<class T, class Alloc = std::allocator<T> >
class BlockingQueue
{
public:
    using Mutex = std::mutex;
    using Lock = std::unique_lock<std::mutex>;
    BlockingQueue(size_t nCap, const Alloc& alloc= Alloc() ): m_alloc(alloc), m_data( m_alloc.allocate(nCap+1) ), m_bufsize(nCap+1)
    {}
    ~BlockingQueue() {
        if( m_data ) {
            clear();
            m_alloc.deallocate( m_data, m_bufsize );
            m_data = nullptr;
        }
    }

    // return false when enqueue fails due to stopped.
    bool enqueue(const T& val) {
        if( m_stopping ) return false;
        Lock locked(m_mut);
        m_condFull.wait( locked, [&]{ return !isFull() || m_stopping.load(); } );

        if( !m_stopping && !isFull() ) {
            pushBack( val );
        }else{ // stopping
            popAll();
            m_condEmpty.notify_one();
            return false;
        }

        locked.unlock();
        m_condEmpty.notify_one();
        return true;
    }

    bool dequeue(T *val) {
        if( m_stopping ) return false;
        Lock locked(m_mut);
        m_condEmpty.wait( locked, [&]{ return !isEmpty() || m_stopping.load();} );

        if( !m_stopping && !isEmpty() ) {
            popFront( val );
        }else{ // stopping
            popAll();
            m_condEmpty.notify_one();
            return false;
        }

        locked.unlock();
        m_condEmpty.notify_one();
        return true;
    }

    void stop() {
        m_stopping = true;
        m_condFull.notify_all();
        m_condEmpty.notify_all();
    }

    void clear() {
        m_stopping = true;
        m_condFull.notify_all();
        m_condEmpty.notify_all();

        {
            Lock locked(m_mut);
            popAll();
        }
    }

    bool empty() {
        Lock locked(m_mut);
        return isEmpty();
    }

    bool stopping() const {
        return m_stopping;
    }

    size_t size() const {
        return m_end > m_begin ? m_end - m_begin : (m_end + m_bufsize - m_begin);
    }


protected:
    bool isFull() const {
        return m_begin == m_end +1 || ( m_begin == 0 && m_end == m_bufsize-1 );  // end - bufsize +1 = begin, when begin and end always increment.
    }
    bool isEmpty() const {
        return m_begin == m_end;
    }
    void pushBack(const T& val) {
        assert( !isFull() );
        new( &m_data[m_end] ) T( val );
        if( ++m_end == m_bufsize ) m_end = 0;
    }
    void popFront(T *val) {
        assert( !isEmpty() );
        if( val ) *val = m_data[m_begin]; // or copy construct
        m_data[m_begin].~T();
        if( ++m_begin == m_bufsize ) m_begin = 0;
    }
    void popAll() {
        while( !isEmpty() ) 
            popFront(nullptr);
    }

    Alloc m_alloc;
    T *m_data = nullptr;
    size_t m_bufsize = 0, m_begin = 0, m_end = 0;
    std::atomic<bool> m_stopping {false};
    Mutex m_mut;
    std::condition_variable m_condFull, m_condEmpty;
};

using EventQueue = SPMCQueue<int>;
//using EventQue = BlockingQueue<int>;

//using Pool = PooledAllocator<Node>;
using Pool = NodePool;
int main(int argc, char *argv[]) 
{
    auto tsStart = std::chrono::steady_clock::now();
    int min_depth = 4;
    int max_depth = std::max(min_depth+2,
                             (argc == 2 ? atoi(argv[1]) : 10));
    int stretch_depth = max_depth+1;

    // Alloc then dealloc stretchdepth tree
    {
        Pool store;
        auto c = make(stretch_depth, store);
        std::cout << "stretch tree of depth " << stretch_depth << "\t "
                  << "check: " << c->check() << std::endl;
    }

    Pool long_lived_store;
    auto long_lived_tree = make(max_depth, long_lived_store);

    // buffer to store output of each thread
    char *outputstr = (char*)malloc(LINE_SIZE * (max_depth +1) * sizeof(char));

    EventQueue que(max_depth);
    std::atomic<int> nEvents{ 0 };
    for (int d = min_depth; d <= max_depth; d += 2 ) {
        if( !que.enqueue(d) ) {
            std::cout << "failed to enque " << nEvents << std::endl;
        }
        ++nEvents;
    }
    if( que.size() != nEvents ) {
        std::cout << "wrong size " << nEvents << que.size() << std::endl;
    }
/*    {
    unsigned int nThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(nThreads);
    for(int j=0; j < nThreads; ++j) {
        threads.push_back( std::thread([&]{ 
                int depth;
                while( que.dequeue(&depth) ) {
                    int iterations = 1 << (max_depth - depth + min_depth);
                    int c = 0;

                    // Create a memory pool for this thread to use.
                    Pool store;

                    for (int i = 1; i <= iterations; ++i) 
                    {
                        auto a = make(depth, store);
                        c += a->check();
//                        store.clear();
                    }

                    // each thread write to separate location
                    sprintf(outputstr + LINE_SIZE * depth, "%d\t trees of depth %d\t check: %d\n",
                       iterations, depth, c);
                    nEvents.fetch_sub(1);
                }
                assert( que.stopping() );
            } ) );
    }
    while( nEvents.load() > 0 )
        ;
    que.clear();
    for(auto& thr: threads ) {
        thr.join();
    }
*/
    #pragma omp parallel for   
    for (int d = min_depth; d <= max_depth; d += 2) 
    {
        int iterations = 1 << (max_depth - d + min_depth);
        int c = 0;

        // Create a memory pool for this thread to use.
        Pool store;

        for (int i = 1; i <= iterations; ++i) 
        {
            auto a = make(d, store);
            c += a->check();
//            store.clear();
        }

        // each thread write to separate location
        sprintf(outputstr + LINE_SIZE * d, "%d\t trees of depth %d\t check: %d\n",
           iterations, d, c);
    }

    auto tsStop = std::chrono::steady_clock::now();
    // print all results
    for (int d = min_depth; d <= max_depth; d += 2) 
        printf("%s", outputstr + (d * LINE_SIZE) );
    free(outputstr);

    std::cout << "long lived tree of depth " << max_depth << "\t "
              << "check: " << (long_lived_tree->check()) << "\n";

    std::cout << "- cpp time(ms): " << (tsStop - tsStart).count()/1000/1000 << std::endl;

    return 0;
}
