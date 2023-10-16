#pragma once

#include <functional>
#include <memory>
#include <format>

template <typename T>
class Susp
{
public:
    Susp() = default;
    explicit Susp(std::function<T()> f)
        : f{ f }, thunk{ &ThunkForce }, memo{ T{} }
    {
        
    }

    T& Get()
    {
        return thunk(this);
    }

    T const& Get() const
    {
        return thunk(this);
    }

    bool IsForced() const
    {
        return thunk == &ThunkGet;
    }

private:
    using ThunkType = T& (*)(Susp*);
    mutable ThunkType thunk;
    mutable T memo;
    std::function<T()> f;

    static T& ThunkForce(Susp* susp)
    {
        return susp->SetMemo();
    }

    static T& ThunkGet(Susp* susp)
    {
        return susp->GetMemo();
    }

    T& GetMemo()
    {
        return memo;
    }

    T const& GetMemo() const
    {
        return memo;
    }

    T& SetMemo()
    {
        memo = f();
        thunk = &ThunkGet;

        return GetMemo();
    }
};

template<class T>
class Stream;

template <typename T>
class Cell
{
public:
    Cell() = default;
    Cell(T v, Stream<T> tail)
        : v{ v }, tail{ std::move(tail) }
    {

    }

    explicit Cell(T v)
        : v{ v }
    {

    }

    T const& Val() const
    {
        return v;
    }

    T& Val()
    {
        return v;
    }

    Stream<T> PoppedFront() const
    {
        return tail;
    }    

private:
    T v;
    Stream<T> tail;
};

template <typename T>
class CellFun
{
public:
    CellFun(T v, Stream<T> s)
        : v{ v }, s{ std::move(s) }
    {

    }

    explicit CellFun(T v)
        : v{ v }
    {

    }

    Cell<T> operator()()
    {
        return Cell<T>{ v, s };
    }

    T v;
    Stream<T> s;
};

template <typename T>
class Stream
{
public:
    using value_type = T;

    Stream() = default;

    explicit Stream(T v)
    {
        auto f{ CellFun<T>{ v } };

        lazy_cell = std::make_shared<Susp<Cell<T>>>(f);
    }

    //template <typename It>
    //Stream(It first, It last)
    //{
    //    if (first != last)
    //    {
    //        auto f{ CellFun<T>{ *first, Stream{ std::next(first), last } } };

    //        lazy_cell = std::make_shared<Susp<Cell<T>>>(f);
    //    }
    //}

    template <typename... Ts>
    requires std::conjunction_v<std::is_same<T, Ts>...>
    Stream(T&& t, Ts&&... ts)
    {
        auto f{ CellFun<T>{ std::forward<T>(t), Stream{ std::forward<Ts>(ts)... } } };

        lazy_cell = std::make_shared<Susp<Cell<T>>>(f);
    }

    Stream(T v, Stream s)
    {
        auto f{ CellFun<T>{ v, std::move(s) } };

        lazy_cell = std::make_shared<Susp<Cell<T>>>(f);
    }

    Stream(std::function<Cell<T>()> f)
        : lazy_cell{ std::make_shared<Susp<Cell<T>>>(f) }
    {

    }

    Stream(Stream const&) noexcept = default;
    Stream(Stream&&) noexcept = default;
    Stream& operator=(Stream const&)noexcept = default;
    Stream& operator=(Stream&&) noexcept = default;

    class Iterator
    {
    public:
        using iterator_concept = std::contiguous_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        Iterator() = default;

        explicit Iterator(Stream<T>* stm)
            : current{ stm }
        {

        }

        T& operator*() const noexcept
        {
            return current->Get();
        }

        Iterator& operator++()
        {
            *current = current->PoppedFront();

            if (current->IsEmpty())
            {
                current = nullptr;
            }

            return *this;
        }

        Iterator& operator++(int)
        {
            auto it{ *this };

            ++(*this);

            return it;
        }

        bool operator==(Iterator const& other) const
        {
            return current == other.current;
        }

        bool operator!=(Iterator const& other) const
        {
            return current != other.current;
        }

    private:
        Stream<T>* current{ nullptr };
    };

    class ConstIterator
    {
    public:
        using iterator_concept = std::contiguous_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = T const*;
        using reference = T const&;

        ConstIterator() = default;

        explicit ConstIterator(Stream<T> const* stm)
            : current{ stm }
        {

        }

        reference operator*() const noexcept
        {
            return current->Get();
        }

        ConstIterator& operator++()
        {
            *const_cast<Stream<T>*>(current) = current->PoppedFront();

            if (current->IsEmpty())
            {
                current = nullptr;
            }

            return *this;
        }

        ConstIterator& operator++(int)
        {
            auto it{ *this };

            ++(*this);

            return it;
        }

        bool operator==(ConstIterator const& other) const
        {
            return current == other.current;
        }

        bool operator!=(ConstIterator const& other) const
        {
            return current != other.current;
        }

    private:
        Stream<T> const* current{ nullptr };
    };

    Iterator begin() noexcept
    {
        return IsEmpty() ? end() : Iterator{ this };
    }

    Iterator end() noexcept
    {
        return Iterator{ nullptr };
    }

    ConstIterator begin() const noexcept
    {
        return IsEmpty() ? end() : ConstIterator{ this };
    }

    ConstIterator end() const noexcept
    {
        return ConstIterator{ nullptr };
    }

    bool IsEmpty() const
    {
        return !lazy_cell;
    }

    T& Get()
    {
        return lazy_cell->Get().Val();
    }

    T const& Get() const
    {
        return lazy_cell->Get().Val();
    }

    Stream PoppedFront() const
    {
        return lazy_cell->Get().PoppedFront();
    }

    bool IsForced() const
    {
        return !IsEmpty() and lazy_cell->IsForced();
    }

    Stream Take(int n) const
    {
        if (n == 0 or IsEmpty())
        {
            return Stream{};
        }

        auto cell{ lazy_cell };

        return Stream{ [cell, n] {
            auto v{ cell->Get().Val() };
            auto t{ cell->Get().PoppedFront() };

            return Cell<T>{ v, t.Take(n - 1) };
        } };
    }

    Stream Drop(int n) const
    {
        if (n == 0)
        {
            return *this;
        }

        if (IsEmpty())
        {
            return Stream{};
        }

        auto t{ PoppedFront() };

        return t.Drop(n - 1);
    }

    Stream Reversed() const
    {
        return Rev(Stream{});
    }

private:
    std::shared_ptr<Susp<Cell<T>>> lazy_cell;

    Stream Rev(Stream acc) const
    {
        if (IsEmpty())
        {
            return acc;
        }

        auto v{ Get() };
        auto t{ PoppedFront() };

        Stream next_acc{ [=] {
            return Cell<T>{ v, acc };
        } };

        return t.Rev(next_acc);
    }
};

Stream<int> IntsFrom(int n)
{
    return Stream<int>{ [n] {
        return Cell<int>{ n, IntsFrom(n + 1) };
    } };
}

Stream<int> Ints(int n, int m)
{
    if (n > m)
    {
        return Stream<int>{};
    }

    return Stream<int>{ [n, m] {
        return Cell<int>(n, Ints(n + 1, m));
    } };
}

template <typename T, typename U, typename F>
auto ZipWith(F f, Stream<T> lhs, Stream<U> rhs) -> Stream<decltype(f(lhs.Get(), rhs.Get()))>
{
    using S = decltype(f(lhs.Get(), rhs.Get()));

    if (lhs.IsEmpty() or rhs.IsEmpty())
    {
        return Stream<S>{};
    }

    return Stream<S>{ [=] {
        return Cell<S>{ f(lhs.Get(), rhs.Get()), ZipWith(f, lhs.PoppedFront(), rhs.PoppedFront()) };
    } };
}

template <typename T, typename F>
void ForEach(Stream<T> strm, F f)
{
    while (!strm.IsEmpty())
    {
        f(strm.Get());

        strm = strm.PoppedFront();
    }
}

template <typename T, typename F>
auto Fmap(Stream<T> stm, F f) -> Stream<decltype(f(stm.Get()))>
{
    using U = decltype(f(stm.Get()));

    if (stm.IsEmpty())
    {
        return Stream<U>{};
    }

    return Stream<U>{ [stm, f] {
        return Cell<U>{ f(stm.Get()), Fmap(stm.PoppedFront(), f) };
    } };
}

template <typename T>
Stream<T> Concat(Stream<T> lhs, Stream<T> rhs)
{
    if (lhs.IsEmpty())
    {
        return rhs;
    }

    return Stream<T>{ [=] {
        return Cell<T>{ lhs.Get(), Concat<T>(lhs.PoppedFront(), rhs) };
    } };
}

template <typename T>
Stream<T> Mjoin(Stream<Stream<T>> stm)
{
    while (!stm.IsEmpty() and stm.Get().IsEmpty())
    {
        stm = stm.PoppedFront();
    }

    if (stm.IsEmpty())
    {
        return Stream<T>{};
    }

    return Stream<T>{ [stm] {
        auto hd{ stm.Get() };

        return Cell<T>{ hd.Get(), Concat(hd.PoppedFront(), Mjoin(stm.PoppedFront())) };
    } };
}

template <typename T, typename F>
auto Mbind(Stream<T> stm, F f) -> decltype(f(stm.Get()))
{
    return Mjoin(Fmap(stm, f));
}

template <typename T, typename F>
auto Fmapv(Stream<T> stm, F f) -> Stream<decltype(f())>
{
    using U = decltype(f());

    if (stm.IsEmpty())
    {
        return Stream<U>{};
    }

    return Stream<U>{ [stm, f] {
        return Cell<U>{ f(), Fmapv(stm.PoppedFront(), f) };
    } };
}

template <typename T, typename F>
auto Mthen(Stream<T> stm, F f) -> decltype(f())
{
    return Mjoin(Fmapv(stm, f));
}

template <typename T>
Stream<T> Mreturn(T v)
{
    return Stream<T>{ [v] {
        return Cell<T>{ v };
    } };
}

Stream<void*> Guard(bool b)
{
    if (b)
    {
        return Stream<void*>{ nullptr };
    }

    return Stream<void*>{};
}

Stream<std::tuple<int, int, int>> Triples()
{
    return Mbind(IntsFrom(1), [](int z) {
        return Mbind(Ints(1, z), [z](int x) {
            return Mbind(Ints(x, z), [x, z](int y) {
                return Mthen(Guard(x * x + y * y == z * z), [x, y, z] {
                    return Mreturn(std::tuple{ x, y, z });
                });
            });
        });
    });
}

template <typename T>
struct std::formatter<Stream<T>, wchar_t>
    : std::formatter<T, wchar_t>
{
    template <typename FormatContext>
    auto format(Stream<T> const& v, FormatContext& ctx) const
    {
        auto out{ ctx.out() };

        if constexpr (std::is_same_v<T, wchar_t>)
        {
            std::ranges::copy(v, out);
        }
        else
        {
            *out++ = L'[';

            for (auto first{ true };
                auto const& e : v)
            {
                if (!std::exchange(first, false))
                {
                    *out++ = L',';
                }

                out = std::format_to(out, L"{}", e);
            }

            *out++ = ']';
        }

        return out;
    }
};