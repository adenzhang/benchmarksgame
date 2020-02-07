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
#include <algorithm>

const size_t LINE_SIZE = 64;

struct DoubleAccumulatedGrowthPolicy
{
    DoubleAccumulatedGrowthPolicy( size_t n ) : m_val( std::max( size_t( 1 ), n ) )
    {
    }

    /// @param toatalVal accumulative total value
    size_t grow_to( size_t toatalVal )
    {
        return std::max( m_val, toatalVal * 2 );
    }
    size_t &get_grow_value()
    {
        return m_val;
    }

private:
    size_t m_val; // current/recent grow value.
};

struct DoublePrevGrowthPolicy : public DoubleAccumulatedGrowthPolicy
{
    DoublePrevGrowthPolicy( size_t n ) : DoubleAccumulatedGrowthPolicy( n )
    {
    }
    size_t grow_to( size_t )
    {
        return ( get_grow_value() *= 2 );
    }
};

struct ConstGrowthPolicy : public DoubleAccumulatedGrowthPolicy
{
    ConstGrowthPolicy( size_t n ) : DoubleAccumulatedGrowthPolicy( n )
    {
    }
    size_t grow_to( size_t )
    {
        return get_grow_value();
    }
};

inline constexpr bool is_pow2( std::size_t n )
{
    return n != 0 && ( n & ( n - 1 ) ) == 0;
}

// Result is >= n
template<class Int>
inline constexpr Int align_up( Int n, Int uiAlignment )
{
    assert( n > 0 );
    assert( is_pow2( uiAlignment ) );
    return ( n + ( uiAlignment - 1 ) ) & ~( uiAlignment - 1 );
}


// @brief FreeList traits:
///    - typename Node
///    - FreeList(FreeList&& )
///    - T *pop()
///    - void push(T &)
///    - bool empty() const
//     - void clear()
template<class T>
struct FreeList
{
    struct FreeNode
    {
        FreeNode *pNext;

        FreeNode( T *next = nullptr ) : pNext( reinterpret_cast<FreeNode *>( next ) )
        {
        }
        FreeNode( FreeNode *next = nullptr ) : pNext( next )
        {
        }
    };
    using Node = FreeNode;

    // static_assert( sizeof( T ) >= sizeof( FreeNode * ), "sizeof T < sizeof(void *)" );

    FreeList() = default;
    FreeList( FreeList &&a )
    {
        pHead = a.pHead;
        a.pHead = nullptr;
    }

    void push( T &p )
    {
        new ( reinterpret_cast<FreeNode *>( &p ) ) FreeNode( pHead );
        pHead = reinterpret_cast<FreeNode *>( &p );
    }

    T *pop()
    {
        T *res = reinterpret_cast<T *>( pHead );
        if ( pHead )
            pHead = pHead->pNext;
        return res;
    }
    bool empty() const
    {
        return !pHead;
    }
    // release head / all.
    void clear()
    {
        pHead = nullptr;
    }
    // FreeList(const FreeList&) = delete;
    FreeNode *pHead = nullptr;
};
/// @brief StaticFnDeleter call fn(D*, T*) to delete object.
template<class T>
struct StaticFnDeleter
{
    using D = void *;
    using DelFn = void ( * )( D, T * );

    DelFn pDelFn = nullptr;
    D pDeleter = D();

    template<class DT>
    StaticFnDeleter( DelFn pF, DT pD ) : pDelFn( pF ), pDeleter( static_cast<D>( pD ) )
    {
    }

    StaticFnDeleter() = default;

    void deallocate( T *p )
    {
        assert( pDelFn );
        ( *pDelFn )( pDeleter, p );
    }

    void operator()( T *p )
    {
        p->~T();
        deallocate( p );
    }
};

// The cache line size is 64 bytes (on modern intel x86_64 processors), but we use 128 bytes to avoid false sharing because the
// prefetcher may read two cache lines.  See section 2.1.5.4 of the intel manual:
// http://www.intel.com/content/dam/doc/manual/64-ia-32-architectures-optimization-manual.pdf
// This is also the value intel uses in TBB to avoid false sharing:
// https://www.threadingbuildingblocks.org/docs/help/reference/memory_allocation/cache_aligned_allocator_cls.htm
// FALSE_SHARING_SIZE 128
template<class T, class GrowthPolicy = DoubleAccumulatedGrowthPolicy, size_t Align = 128>
class ChunkAllocator : protected GrowthPolicy
{
    // memory layout : ChunckInfo| slot1 | slot2 | ... |
    // size of ChunckInfo is same as size of slot.
    struct ChunkInfo;

    struct Slot
    {
        // unsigned m_index;
        T m_data;

        T &getData()
        {
            return m_data;
        }
    };

    struct ChunkInfo
    {
        ChunkInfo *m_pNext = nullptr;
        unsigned m_cap = 0; // total slots
        unsigned m_size = 0; // used slots

        void init( unsigned cap )
        {
            m_cap = cap;
            m_size = 0;
            m_pNext = nullptr;
        }
        void clear()
        {
            m_size = 0;
        }
        // return updated size
        unsigned incSize()
        {
            return m_cap == m_size ? 0 : ++m_size;
        }

        bool full() const
        {
            return m_cap == m_size;
        }

        void addNext( ChunkInfo *p )
        {
            m_pNext = p;
        }
    };
    static constexpr size_t ChunkInfoSize = align_up( sizeof( ChunkInfo ), Align );
    static constexpr size_t SlotSize = align_up( sizeof( Slot ), Align );
    static constexpr bool IsPOD = std::is_pod<T>::value;
    static constexpr size_t BoundarySize = 1 << 12; // 4k page size

    static Slot &getSlot( ChunkInfo &info, size_t k )
    {
        return *reinterpret_cast<Slot *>( reinterpret_cast<char *>( &info ) + ChunkInfoSize + SlotSize * k );
    }
    // static ChunkInfo& getChunkInfo(Slot& slot){
    //     return *static_cast<ChunkInfo*>( reinterpret_cast<char*>(&info)  - slot.m_index * SlotSize - ChunkInfoSize);
    // }

    static Slot *allocate_slot( ChunkInfo &info )
    {
        return info.incSize() ? &getSlot( info, info.m_size - 1 ) : nullptr;
    }

    static ChunkInfo *allocate_chunk( size_t minSlots )
    {
        assert( minSlots );
        size_t nbytes = ChunkInfoSize + SlotSize * minSlots;
        nbytes = align_up( nbytes, BoundarySize );
        auto pChunk = static_cast<ChunkInfo *>( std::aligned_alloc( Align, nbytes ) );
        if ( pChunk )
        {
            pChunk->init( ( nbytes - ChunkInfoSize ) / SlotSize );
        }
        // std::cout << "alloc addr: " << pChunk << ", slots:" << pChunk->m_cap << ", bytes:" << nbytes << std::endl;
        assert( pChunk );
        return pChunk;
    }

public:
    ChunkAllocator( size_t nReserve = 0, size_t growthParam = 0 ) : GrowthPolicy( growthParam ? growthParam : nReserve )
    {
        if ( nReserve > 0 )
            add_chunk();
    }

    // copy constructor only copies parameters, not memory.
    ChunkAllocator( const ChunkAllocator &a ) : ChunkAllocator( a.m_totalSlots, a.get_grow_value() )
    {
    }
    ChunkAllocator( ChunkAllocator &&a ) : GrowthPolicy( a.get_grow_value() ), m_totalSlots( a.m_totalSlots ), m_pChunkList( a.m_pChunkList )
    {
        a.m_pChunkList = nullptr;
    }
    ~ChunkAllocator()
    {
        destroy();
    }

    T *allocate( size_t = 1 )
    {
        if ( !m_pChunkList || m_pChunkList->full() )
            add_chunk();
        assert( m_pChunkList );
        assert( !m_pChunkList->full() );
        return &allocate_slot( *m_pChunkList )->getData();
    }

    /// @brief recycle, mark all allocated slot un-allocated.
    void clear()
    {
        for ( auto pChunk = m_pChunkList; pChunk; pChunk = pChunk->m_pNext )
            pChunk->clear();
    }
    // destroy all, but no deallocate provided.
    void destroy()
    {
        while ( m_pChunkList )
        {
            auto p = m_pChunkList;
            m_pChunkList = m_pChunkList->m_pNext;
            // std::cout << "free addr:" << p << ", slots:" << p->m_cap << std::endl;
            std::free( p );
        }
        m_totalSlots = 0;
    }

protected:
    void add_chunk()
    {
        auto pChunk = allocate_chunk( GrowthPolicy::grow_to( m_totalSlots ) );
        pChunk->addNext( m_pChunkList );
        m_pChunkList = pChunk;
        m_totalSlots += m_pChunkList->m_cap;
        GrowthPolicy::get_grow_value() = m_pChunkList->m_cap;
    }
    size_t m_totalSlots = 0;
    ChunkInfo *m_pChunkList = nullptr;
};

template<class T, class Alloc = ChunkAllocator<T>, class FreeList = FreeList<T>>
class PooledAllocator : private Alloc
{
public:
    // using Alloc = typename std::allocator_traits<AllocT>::template rebind_alloc<typename FreeList::Node>;
    PooledAllocator( size_t nReserve = 0, Alloc &&alloc = Alloc() ) : Alloc( std::move( alloc ) )
    {
        for ( size_t i = 0; i < nReserve; ++i )
        {
            if ( auto p = reinterpret_cast<T *>( get_allocator().allocate( 1 ) ) )
            {
                ++m_nAllocated;
                deallocate( p );
            }
            else
                break;
        }
    }
    ~PooledAllocator()
    {
        // if ( m_nAllocated != m_nFree )
        //     std::cout << "allocated: " << m_nAllocated << ", in FreeList: " << m_nFree << std::endl;
        // assert( m_nAllocated == m_nFree );
    }

    T *allocate()
    {
        auto p = m_freeList.pop();
        if ( p )
        {
            --m_nFree;
            return p;
        }
        ++m_nAllocated; // note: allocate may fail.
        return reinterpret_cast<T *>( get_allocator().allocate( 1 ) );
    }

    template<class... Args>
    std::unique_ptr<T, StaticFnDeleter<T>> create_unique( Args &&... args )
    {
        auto p = allocate();
        new ( p ) T( std::forward<Args>( args )... );
        return std::unique_ptr<T, StaticFnDeleter<T>>( p, get_deleter() );
    }

    template<class... Args>
    T *create( Args &&... args )
    {
        auto p = allocate();
        new ( p ) T( std::forward<Args>( args )... );
        return p;
    }

    void deallocate( T *p, size_t = 1 )
    {
        m_freeList.push( *p );
        ++m_nFree;
    }

    static void Delete( void *self, T *p )
    {
        reinterpret_cast<PooledAllocator *>( self )->deallocate( p );
    }

    void clear()
    {
        get_allocator().clear();
        m_freeList.clear();
    }

    StaticFnDeleter<T> get_deleter()
    {
        return StaticFnDeleter<T>( Delete, this );
    }

    Alloc &get_allocator()
    {
        return static_cast<Alloc &>( *this );
    }

protected:
    FreeList m_freeList;
    size_t m_nAllocated = 0;
    size_t m_nFree = 0;
};

struct Node
{
    using Deleter = StaticFnDeleter<Node>;
    using Ptr = Node *; // std::unique_ptr<Node, Deleter>;

    Ptr l;
    Ptr r;

    Node() = default;

    int check() const
    {
        if ( l )
            return l->check() + 1 + r->check();
        else
            return 1;
    }
};
struct Node0
{
    Node0 *l = nullptr, *r = nullptr;

    int check() const
    {
        if ( l )
            return l->check() + 1 + r->check();
        else
            return 1;
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

    Node0 *allocate()
    {
        return static_cast<Node0 *>( malloc( sizeof( Node ) ) );
        //        return (Node *)apr_palloc(pool, sizeof(Node));
    }

    void clear()
    {
        //        apr_pool_clear(pool);
    }

private:
    //    apr_pool_t* pool;
};

Node0 *make( int d, NodePool &store )
{
    Node0 *root = store.allocate();

    if ( d > 0 )
    {
        root->l = make( d - 1, store );
        root->r = make( d - 1, store );
    }

    return root;
}

NodePtr make( int d, PooledAllocator<Node> &store )
{
    //    Node* root = store.allocate();
    auto root = store.create(); // store.create_unique();

    if ( d > 0 )
    {
        root->l = make( d - 1, store );
        root->r = make( d - 1, store );
    }

    return root;
}


template<class T, class AllocT = std::allocator<T>>
class SPMCQueue
{
    struct Node
    {
        std::atomic<bool> flag; // true for having value
        T val;
    };

public:
    using Alloc = typename std::allocator_traits<AllocT>::template rebind_alloc<Node>;

    SPMCQueue( size_t nCap, const Alloc &alloc = Alloc() ) : m_alloc( alloc ), m_data( m_alloc.allocate( nCap + 1 ) ), m_bufsize( nCap + 1 )
    {
        for ( size_t i = 0; i < m_bufsize; ++i )
            m_data[i].flag.store( false );
    }
    ~SPMCQueue()
    {
        if ( m_data )
        {
            clear();
            m_alloc.deallocate( m_data, m_bufsize );
            m_data = nullptr;
        }
    }

    /// @return false when full. Note: the front element mayb be in process of dequeue.
    bool enqueue( const T &val )
    {
        auto iEnd = m_end.load( std::memory_order_acquire ) % m_bufsize;
        auto iNext = ( iEnd + 1 ) % m_bufsize;
        if ( m_data[iNext].flag.load( std::memory_order_acquire ) ) // full
            return false;
        assert( m_data[iEnd].flag.load( std::memory_order_acquire ) == false );
        m_data[iEnd].val = val;
        assert( !m_data[iEnd].flag );
        m_data[iEnd].flag.store( true, std::memory_order_release );
        m_end.fetch_add( 1 );
        return true;
    }

    bool dequeue( T *val )
    {
        size_t ibegin = m_begin.load( std::memory_order_acquire );
        int n = 0;
        do
        {
            if ( ibegin == m_end.load( std::memory_order_acquire ) ) // emtpy
                return false;
        } while ( !m_begin.compare_exchange_weak( ibegin, ibegin + 1 ) ); // std::memory_order_acquire

        // alreay acquired the ownership.
        ibegin %= m_bufsize;
        auto &data = m_data[ibegin];
        assert( data.flag.load( std::memory_order_acquire ) );

        if ( val )
            new ( val ) T( std::move( data.val ) );
        data.val.~T();
        data.flag.store( false, std::memory_order_release );
        return true;
    }

    size_t size() const
    {
        return m_end.load() - m_begin.load();
    }

    bool empty() const
    {
        return m_begin.load() == m_end.load();
    }

    /// weak full. dequeue may be in process.
    bool full() const
    {
        return m_begin.load() + ( m_bufsize - 1 ) == m_end.load();
    }

    void stop()
    {
        m_stopping = true;
    }

    bool stopping() const
    {
        return m_stopping;
    }

    void clear()
    {
        while ( dequeue( nullptr ) )
            ;
    }

protected:
    Alloc m_alloc;
    Node *m_data = nullptr;
    size_t m_bufsize = 0;
    std::atomic<size_t> m_begin{0}, m_end{0};
    std::atomic<bool> m_stopping{false};
};

template<class T, class Alloc = std::allocator<T>>
class BlockingQueue
{
public:
    using Mutex = std::mutex;
    using Lock = std::unique_lock<std::mutex>;
    BlockingQueue( size_t nCap, const Alloc &alloc = Alloc() ) : m_alloc( alloc ), m_data( m_alloc.allocate( nCap + 1 ) ), m_bufsize( nCap + 1 )
    {
    }
    ~BlockingQueue()
    {
        if ( m_data )
        {
            clear();
            m_alloc.deallocate( m_data, m_bufsize );
            m_data = nullptr;
        }
    }

    // return false when enqueue fails due to stopped.
    bool enqueue( const T &val )
    {
        if ( m_stopping )
            return false;
        Lock locked( m_mut );
        m_condFull.wait( locked, [&] { return !isFull() || m_stopping.load(); } );

        if ( !m_stopping && !isFull() )
        {
            pushBack( val );
        }
        else
        { // stopping
            popAll();
            m_condEmpty.notify_one();
            return false;
        }

        locked.unlock();
        m_condEmpty.notify_one();
        return true;
    }

    bool dequeue( T *val )
    {
        if ( m_stopping )
            return false;
        Lock locked( m_mut );
        m_condEmpty.wait( locked, [&] { return !isEmpty() || m_stopping.load(); } );

        if ( !m_stopping && !isEmpty() )
        {
            popFront( val );
        }
        else
        { // stopping
            popAll();
            m_condEmpty.notify_one();
            return false;
        }

        locked.unlock();
        m_condEmpty.notify_one();
        return true;
    }

    void stop()
    {
        m_stopping = true;
        m_condFull.notify_all();
        m_condEmpty.notify_all();
    }

    void clear()
    {
        m_stopping = true;
        m_condFull.notify_all();
        m_condEmpty.notify_all();

        {
            Lock locked( m_mut );
            popAll();
        }
    }

    bool empty()
    {
        Lock locked( m_mut );
        return isEmpty();
    }

    bool stopping() const
    {
        return m_stopping;
    }

    size_t size() const
    {
        return m_end > m_begin ? m_end - m_begin : ( m_end + m_bufsize - m_begin );
    }


protected:
    bool isFull() const
    {
        return m_begin == m_end + 1 || ( m_begin == 0 && m_end == m_bufsize - 1 ); // end - bufsize +1 = begin, when begin and end always increment.
    }
    bool isEmpty() const
    {
        return m_begin == m_end;
    }
    void pushBack( const T &val )
    {
        assert( !isFull() );
        new ( &m_data[m_end] ) T( val );
        if ( ++m_end == m_bufsize )
            m_end = 0;
    }
    void popFront( T *val )
    {
        assert( !isEmpty() );
        if ( val )
            *val = m_data[m_begin]; // or copy construct
        m_data[m_begin].~T();
        if ( ++m_begin == m_bufsize )
            m_begin = 0;
    }
    void popAll()
    {
        while ( !isEmpty() )
            popFront( nullptr );
    }

    Alloc m_alloc;
    T *m_data = nullptr;
    size_t m_bufsize = 0, m_begin = 0, m_end = 0;
    std::atomic<bool> m_stopping{false};
    Mutex m_mut;
    std::condition_variable m_condFull, m_condEmpty;
};

using EventQueue = SPMCQueue<int>;
// using EventQueue = BlockingQueue<int>;

using Pool = PooledAllocator<Node>;
// using Pool = NodePool;
int main( int argc, char *argv[] )
{
    auto tsStart = std::chrono::steady_clock::now();
    int min_depth = 4;
    int max_depth = std::max( min_depth + 2, ( argc == 2 ? atoi( argv[1] ) : 10 ) );
    int stretch_depth = max_depth + 1;

    // Alloc then dealloc stretchdepth tree
    {
        Pool store;
        auto c = make( stretch_depth, store );
        std::cout << "stretch tree of depth " << stretch_depth << "\t "
                  << "check: " << c->check() << std::endl;
    }

    Pool long_lived_store;
    auto long_lived_tree = make( max_depth, long_lived_store );

    // buffer to store output of each thread
    char *outputstr = (char *)malloc( LINE_SIZE * ( max_depth + 1 ) * sizeof( char ) );

    EventQueue que( max_depth );
    std::atomic<int> nEvents{0};
    for ( int d = min_depth; d <= max_depth; d += 2 )
    {
        if ( !que.enqueue( d ) )
        {
            std::cout << "failed to enque " << nEvents << std::endl;
        }
        ++nEvents;
    }
    if ( que.size() != nEvents )
    {
        std::cout << "wrong size " << nEvents << que.size() << std::endl;
    }

    unsigned int nThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads( nThreads );
    for ( int j = 0; j < nThreads; ++j )
    {
        threads[j] = std::thread( [&] {
            int depth;
            while ( que.dequeue( &depth ) )
            {
                int iterations = 1 << ( max_depth - depth + min_depth );
                int c = 0;

                // Create a memory pool for this thread to use.
                Pool store;

                for ( int i = 1; i <= iterations; ++i )
                {
                    auto a = make( depth, store );
                    c += a->check();
                    store.clear();
                }

                // each thread write to separate location
                sprintf( outputstr + LINE_SIZE * depth, "%d\t trees of depth %d\t check: %d\n", iterations, depth, c );
                nEvents.fetch_sub( 1 );
            }
            // assert( que.stopping() );
        } );
    }
    while ( nEvents.load() > 0 )
        ;
    que.clear();
    for ( auto &thr : threads )
    {
        thr.join();
    }

    /*
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
*/
    auto tsStop = std::chrono::steady_clock::now();
    // print all results
    for ( int d = min_depth; d <= max_depth; d += 2 )
        printf( "%s", outputstr + ( d * LINE_SIZE ) );
    free( outputstr );

    std::cout << "long lived tree of depth " << max_depth << "\t "
              << "check: " << ( long_lived_tree->check() ) << "\n";

    std::cout << "- cpp time(ms): " << ( tsStop - tsStart ).count() / 1000 / 1000 << std::endl;

    return 0;
}
