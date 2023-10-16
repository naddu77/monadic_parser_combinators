export module Parser;

export import Stream;
import std;

using namespace std::literals;
using namespace StreamModule;

export namespace ParserModule
{
	template <typename A>
    class Parser
    {
    public:
        using value_type = A;
        using Func = std::function<Stream<std::tuple<A, std::wstring_view>>(std::wstring_view)>;

        static auto Make(Func&& func)
        {
            return Parser<A>{ std::forward<Func>(func) };
        }

        Parser() = default;

        Parser(Func&& op)
            : op{ std::forward<Func>(op) }
        {

        }

        Parser(Parser const&) = default;
        Parser(Parser&&) = default;
        Parser& operator=(Parser const&) = default;
        Parser& operator=(Parser&&) = default;

        Stream<std::tuple<A, std::wstring_view>> operator()(std::wstring_view inp) const
        {
            return op ? op(inp) : Stream<std::tuple<A, std::wstring_view>>{};
        }

    private:
        Func op;
    };

    template <typename A>
    void Show(Stream<std::tuple<A, std::wstring_view>> v)
    {
        std::wcout << '[';

        for (auto const& e : v
            | std::views::transform([]<typename T>(T const& t) {
                if constexpr (std::disjunction_v<std::is_constructible<std::wstring_view, std::tuple_element_t<0, T>>, std::is_same<Stream<wchar_t>, std::tuple_element_t<0, T>>>)
                {
                    return std::format(L"(\"{}\",\"{}\")", std::get<0>(t), std::get<1>(t));
                }
                else
                {
                    return std::format(L"({},\"{}\")", std::get<0>(t), std::get<1>(t));
                }
            })
            | std::views::join_with(L","s)
        )
        {
            std::wcout << e;
        }

        std::wcout << L"]\n";
    }

    template <typename A>
    Parser<A> Result(A v)
    {
        return Parser<A>::Make([v](std::wstring_view inp) {
            return Stream<std::tuple<A, std::wstring_view>>{ { v, inp } };
        });
    }

    template <typename A>
    Parser<A> Zero()
    {
        return Parser<A>::Make([](std::wstring_view) -> Stream<std::tuple<A, std::wstring_view>> {
            return {};
        });
    }

    template <typename A, typename Func>
    auto Bind(Parser<A> p, Func f)
    {
        using B = typename std::invoke_result_t<Func, A>::value_type;

        return Parser<B>::Make([p, f](std::wstring_view inp) -> Stream<std::tuple<B, std::wstring_view>> {
            return p(inp) >>= [f](std::tuple<A, std::wstring_view> t1) {
                auto [v, inp1] { t1 };

                return f(v)(inp1) >>= [](std::tuple<B, std::wstring_view> t2) {
                    return Mreturn(t2);
                };
            };
        });
    }

    template <typename M, typename F>
    constexpr auto operator>>=(M&& m, F&& f)
    {
        return Bind(std::forward<M>(m), std::forward<F>(f));
    }

    template <typename A, typename B>
    constexpr auto operator>>(Parser<A> pa, Parser<B> pb)
    {
        return pa >>= [pb](A a) {
            return pb;
        };
    }

    template <typename A>
    Parser<A> Plus(Parser<A> pa1, Parser<A> pa2)
    {
        return Parser<A>::Make([pa1, pa2](std::wstring_view inp) -> Stream<std::tuple<A, std::wstring_view>> {
            return PlusPlus<std::tuple<A, std::wstring_view>>(pa1(inp), pa2(inp));
        });
    }
}
