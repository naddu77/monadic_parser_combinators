//import std;
#include "LazyStream.h"
#include <ranges>
#include <string_view>
#include <format>
#include <print>
#include <vector>
#include <functional>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <variant>
#include <memory>
#include "generator.h"

using namespace std::literals;

template <typename A>
class Parser
{
public:
    using Func = std::function<Stream<std::tuple<A, std::wstring_view>>(std::wstring_view)>;

    static auto Make(Func&& func)
    {
        return Parser<A>{ std::forward<Func>(func) };
    }

    Parser() = default;

    Parser(Func&& op)
        : op{ std::forward<Func>(op) }
    {
        if (this->op == nullptr)
        {
            assert(false);
        }
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

template <typename B, typename A>
using ParserLambda = std::function<Parser<B>(A)>;

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
    return Parser<A>::Make([v](std::wstring_view inp) -> Stream<std::tuple<A, std::wstring_view>> {
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

auto Item()
{
    return Parser<wchar_t>::Make([](std::wstring_view inp) -> Stream<std::tuple<wchar_t, std::wstring_view>> {
        if (std::empty(inp))
        {
            return Stream<std::tuple<wchar_t, std::wstring_view>>{};
        }

        auto x{ inp.front() };

        inp.remove_prefix(1);

        return Stream<std::tuple<wchar_t, std::wstring_view>>{ std::tuple{ x, inp } };
    });
}

template <typename A, typename B>
auto Seq(Parser<A> pa, Parser<B> pb)
{
    return Parser<std::tuple<A, B>>::Make([pa, pb](std::wstring_view inp) -> Stream<std::tuple<std::tuple<A, B>, std::wstring_view>> {
        return Mbind(pa(inp), [pb](std::tuple<A, std::wstring_view> t1) {
            auto const& [v, inp1] { t1 };

            return Mbind(pb(inp1), [](std::tuple<B, std::wstring_view> t2) {
                auto const& [w, inp2] { t2 };

                return Mreturn(std::tuple{ std::tuple{ v, w }, inp2 });
            });
        });
    });
}

template <typename A, typename B>
auto Bind(Parser<A> p, std::function<Parser<B>(A)> f)
{
    return Parser<B>::Make([p, f](std::wstring_view inp) mutable -> Stream<std::tuple<B, std::wstring_view>> {
        return Mbind(p(inp), [f](std::tuple<A, std::wstring_view> t1) {
            auto [v, inp1] { t1 };

            return Mbind(f(v)(inp1), [](std::tuple<B, std::wstring_view> t2) {
                return Mreturn(t2);
            });
        });
    });
}

auto Sat(std::function<bool(wchar_t)> p)
{
    return Bind(Item(), ParserLambda<wchar_t, wchar_t>{ [p](wchar_t x) mutable {
        return p(x) ? Result(x) : Zero<wchar_t>();
    } });
}

auto Char(wchar_t x)
{
    return Sat([x](wchar_t y) { return x == y; });
}

auto Digit()
{
    return Sat([](wchar_t x) { return L'0' <= x and x <= L'9'; });
}

auto Lower()
{
    return Sat([](wchar_t x) { return L'a' <= x and x <= L'z'; });
}

auto Upper()
{
    return Sat([](wchar_t x) { return L'A' <= x and x <= L'Z'; });
}

template <typename T>
Stream<T> PlusPlus(Stream<T> stm1, Stream<T> stm2)
{
    return Concat(stm1, stm2);
}

template <typename A>
Parser<A> Plus(Parser<A> pa1, Parser<A> pa2)
{
    return Parser<A>::Make([pa1, pa2](std::wstring_view inp) mutable -> Stream<std::tuple<A, std::wstring_view>> {
        return PlusPlus<std::tuple<A, std::wstring_view>>(pa1(inp), pa2(inp));
    });
}

auto Letter()
{
    return Plus(Lower(), Upper());
}

auto AlphaNum()
{
    return Plus(Letter(), Digit());
}

Parser<std::wstring> Word()
{
    auto ne_word{ Bind(Letter(), ParserLambda<std::wstring, wchar_t>{ [](wchar_t x) {
        return Bind(Word(), ParserLambda<std::wstring, std::wstring>{ [x](std::wstring xs) {
            return Result(std::wstring{ x } + xs);
        } });
    } }) };

    return Plus(ne_word, Result(L""s));
}

Parser<std::wstring> String(std::wstring str)
{
    if (std::empty(str))
    {
        return Result(L""s);
    }

    auto x{ str.front() };
    auto xs{ str.substr(1) };

    return Bind(Char(x), ParserLambda<std::wstring, wchar_t>{ [x, xs](wchar_t x) {
        return Bind(String(xs), ParserLambda<std::wstring, std::wstring>{ [x, xs](std::wstring) {
            return Result(std::wstring{ x } + xs);
        } });
    } });
}

template <typename A>
Parser<Stream<A>> Many(Parser<A> p);

//template <typename A>
//Parser<Stream<A>> Many(Parser<A> p)
//{
//    auto ne_many{ Bind(p, ParserLambda<Stream<A>, A>{ [p](A x) {
//        return Bind(Many(p), ParserLambda<Stream<A>, Stream<A>>{ [x](Stream<A> xs) {
//            return Result(Stream<A>{ x, xs });
//        } });
//    } }) };
//
//    return Plus(ne_many, Result(Stream<A>{}));
//}

Parser<Stream<wchar_t>> Ident()
{
    return Bind(Lower(), ParserLambda<Stream<wchar_t>, wchar_t>{ [](wchar_t x) {
        return Bind(Many(AlphaNum()), ParserLambda<Stream<wchar_t>, Stream<wchar_t>>{ [x](Stream<wchar_t> xs) {
            return Result(Stream<wchar_t>{ x, xs });
        } });
    } });
}

template <typename A>
Parser<Stream<A>> Many1(Parser<A> p)
{
    return Bind(p, ParserLambda<Stream<A>, A>{ [p](A x) {
        return Bind(Many(p), ParserLambda<Stream<A>, Stream<A>>{ [x](Stream<A> xs) {
            return Result(Stream<A>{ x, xs });
        } });
    } });
}

template <typename Func, typename T>
auto Foldl1(Func const& f, Stream<T> const& xs)
{
    T result{};

    while (!xs.IsEmpty())
    {
        result = f(result, xs.Get());

        xs = xs.PoppedFront();
    }

    return result;
}

template <typename Func, typename T>
auto Foldl(Func const& f, T const& x, Stream<T> const& xs)
{
    T result{ x };

    while (!xs.IsEmpty())
    {
        result = f(result, xs.Get());

        xs = xs.PoppedFront();
    }

    return result;
}

Parser<int> Nat();

Parser<int> Int()
{
    return Plus(
        Bind(Char(L'-'), ParserLambda<int, wchar_t>{ [](wchar_t) {
            return Bind(Nat(), ParserLambda<int, int>{ [](int n) {
                return Result(-n);
                } });
            } }),
        Nat()
    );
}

template <typename A, typename B>
Parser<Stream<A>> SepBy1(Parser<A> p, Parser<B> sep)
{
    return Bind(p, ParserLambda<Stream<A>, A>{ [p, sep](A x) {
        auto many{
            Many(
                Bind(sep, ParserLambda<A, B>{ [p](B) {
                    return Bind(p, ParserLambda<A, A>{ [](A y) {
                        return Result(y);
                    } });
                } })
            )
        };

        return Bind(many, ParserLambda<Stream<A>, Stream<A>>{ [x](Stream<A> xs) {
            return Result(Stream<A>{ x, xs });
        }});
    } });
}

template <typename A, typename B, typename C>
Parser<B> Bracket(Parser<A> open, Parser<B> p, Parser<C> close)
{
    return Bind(open, ParserLambda<B, A>{ [p, close](A) {
        return Bind(p, ParserLambda<B, B>{ [close](B x) {
            return Bind(close, ParserLambda<B, C>{ [x](C) {
                return Result(x);
            } });
        } });
    } });
}

Parser<Stream<int>> Ints()
{
    return Bracket(Char(L'['), SepBy1(Int(), Char(L',')), Char(L']'));
}

template <typename A, typename B>
Parser<Stream<A>> SepBy(Parser<A> p, Parser<B> sep)
{
    return Plus(SepBy1(p, sep), Result(Stream<A>{}));
}

Parser<int> Factor(int stack_overflow);
Parser<std::function<int(int, int)>> AddOp();

template <typename A>
Parser<A> Chainl1(Parser<A> p, Parser<std::function<A(A, A)>> const& op);

Parser<int> Expr(int stack_overflow = 10);

Parser<int> Factor(int stack_overflow)
{
    return Plus(Nat(), Bracket(Char(L'('), Expr(stack_overflow - 1), Char(L')')));
}

template <typename Func, typename T>
auto Foldr1(Func const& f, Stream<T> const& xs)
{
    T result{};
    auto reversed_xs{ xs.Reversed() };

    while (!reversed_xs.IsEmpty())
    {
        result = f(reversed_xs.Get(), result);

        reversed_xs = reversed_xs.PoppedFront();
    }

    return result;
}

template <typename A, typename B>
Parser<B> Ops(Stream<std::tuple<Parser<A>, B>> xs)
{
    auto result{
        Fmap(xs, [xs](std::tuple<Parser<A>, B> t) {
            auto const& [p, op] { t };

            return Bind(p, ParserLambda<B, A>{ [op](A) {
                return Result(op);
            } });
        })
    };

    return Foldr1(&Plus<B>, result);
}

Parser<std::function<int(int, int)>> AddOp()
{
    return Ops(Stream<std::tuple<Parser<wchar_t>, std::function<int(int, int)>>>{ 
        std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'+'), std::plus<int>{} },
        std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'-'), std::minus<int>{} }
    });
}

template <typename A>
std::function<Parser<A>(A)> Rest(Parser<A> p, Parser<std::function<A(A, A)>> const& op)
{
    return [p, op](A x) {
        return Plus(
            Bind(op, ParserLambda<A, std::function<A(A, A)>>{ [p, op, x](std::function<A(A, A)> const& f) {
                return Bind(p, ParserLambda<A, A>{ [p, op, x, f](A y) {
                    return Rest(p, op)(f(x, y));
                } });
            } }),
            Result(x)
        );
    };
}

template <typename A>
Parser<A> Chainl1(Parser<A> p, Parser<std::function<A(A, A)>> const& op)
{
    return Bind(p, Rest(p, op));
}

Parser<int> Nat()
{
    std::function<int(int, int)> op{ [](int m, int n) { return 10 * m + n; } };

    return Chainl1(
        Bind(Digit(), ParserLambda<int, wchar_t>{ [](wchar_t x) {
            return Result(x - L'0');
            } }),
        Result(op)
    );
}

template <typename A>
Parser<A> Chainr1(Parser<A> p, Parser<std::function<A(A, A)>> const& op)
{
    return Bind(p, ParserLambda<A, A>{ [p, op](A x) {
        return Plus(
            Bind(op, ParserLambda<A, std::function<A(A, A)>>{ [p, op, x](std::function<A(A, A)> const& f) {
                return Bind(Chainr1(p, op), ParserLambda<A, A>{ [f, x](A y) {
                    return Result(f(x, y));
                } });
            } }),
            Result(x)
        );
    } });
}

Parser<int> Term(int stack_overflow);
Parser<std::function<int(int, int)>> ExpOp();

Parser<int> Expr(int so)
{
    if (so <= 0)
    {
        return Result(0);
    }

    return Chainl1(Term(so - 1), AddOp());
}


Parser<int> Term(int stack_overflow)
{
    return Chainr1(Factor(stack_overflow), ExpOp());
}

Parser<std::function<int(int, int)>> ExpOp()
{
    return Ops(Stream<std::tuple<Parser<wchar_t>, std::function<int(int, int)>>>{ 
        std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'^'), [](int x, int y) { return static_cast<int>(std::pow(x, y)); } }
    });
}

template <typename A>
Parser<A> Chainl(Parser<A> p, Parser<std::function<A(A, A)>> const& op, A v)
{
    return Plus(Chainl1(p, op), Result(v));
}

template <typename A>
Parser<A> Chainr(Parser<A> p, Parser<std::function<A(A, A)>> const& op, A v)
{
    return Plus(Chainr1(p, op), Result(v));
}

template <typename A, typename B>
A Fst(std::tuple<A, B> t)
{
    return std::get<0>(t);
}

template <typename A, typename B>
B Snd(std::tuple<A, B> t)
{
    return std::get<1>(t);
}

template <typename A>
A Head(Stream<A> xs)
{
    return xs.Get();
}

template <typename A>
Stream<A> Tail(Stream<A> xs)
{
    return xs.PoppedFront();
}

template <typename A>
Parser<A> Force(Parser<A> p)
{
    return Parser<A>::Make([p](std::wstring_view inp) -> Stream<std::tuple<A, std::wstring_view>> {
        auto x{ p(inp) };

        if (x.IsEmpty())
        {
            return {};
        }

        return { std::tuple<A, std::wstring_view>{ Fst(Head(x)), Snd(Head(x)) }, Tail(x) };
    });
}

template <typename A>
Parser<Stream<A>> Many(Parser<A> p)
{
    return Force(Plus(
        Bind(p, ParserLambda<Stream<A>, A>{ [p](A x) {
            return Bind(Many(p), ParserLambda<Stream<A>, Stream<A>>{ [x](Stream<A> xs) {
                return Result(Stream<A>{ x, xs });
            } });
        } }),
        Result(Stream<A>{})
    ));
}

Parser<int> Number()
{
    return Plus(Nat(), Result(0));
}

template <typename A>
Parser<A> First(Parser<A> p);

template <typename A>
Parser<A> PlusPlusPlus(Parser<A> p, Parser<A> q)
{
    return First(Plus(p, q));
}

Parser<std::wstring> Colour()
{
    return PlusPlusPlus(String(L"yello"), String(L"orange"));
}

template <typename A>
Parser<A> First(Parser<A> p)
{
    return Parser<A>{ [p](std::wstring_view inp) {
        auto result{ p(inp) };

        if (result.IsEmpty())
        {
            return result;
        }

        return result.Take(1);
    } };
}

Parser<std::tuple<>> Spaces()
{
    auto is_space{ [](wchar_t x) { return x == L' ' or x == L'\n' or x == L'\t'; } };

    return Bind(Many1(Sat(is_space)), ParserLambda<std::tuple<>, Stream<wchar_t>>{ [](Stream<wchar_t>) {
        return Result(std::tuple<>{});
    } });
}

Parser<std::tuple<>> Comment()
{
    return Bind(String(L"--"), ParserLambda<std::tuple<>, std::wstring>{ [](std::wstring) {
        return Bind(Many(Sat([](wchar_t x) { return x != L'\n'; })), ParserLambda<std::tuple<>, Stream<wchar_t>>{ [](Stream<wchar_t>) {
            return Result(std::tuple<>{});
        } });
    } });
}

Parser<std::tuple<>> Junk()
{
    return Bind(Many(PlusPlusPlus(Spaces(), Comment())), ParserLambda<std::tuple<>, Stream<std::tuple<>>>{ [](Stream<std::tuple<>>) {
        return Result(std::tuple<>{});
    } });
}

template <typename A>
Parser<A> Parse(Parser<A> p)
{
    return Bind(Junk(), ParserLambda<A, std::tuple<>>{ [](std::tuple<>) {
        return Bind(p, ParserLambda<A, A>{ [](A v) {
            return Result(v);
        } });
    } });
}

template <typename A>
Parser<A> Token(Parser<A> p)
{
    return Bind(p, ParserLambda<A, A>{ [](A v) {
        return Bind(Junk(), ParserLambda<A, std::tuple<>>{ [v](std::tuple<>) {
            return Result(v);
        } });
    } });
}

Parser<int> Natural()
{
    return Token(Nat());
}

Parser<int> Integer()
{
    return Token(Int());
}

Parser<std::wstring> Symbol(std::wstring xs)
{
    return Token(String(xs));
}

Parser<std::wstring> Identifier(Stream<std::wstring> ks)
{
    return Token(
        Bind(Ident(), ParserLambda<std::wstring, Stream<wchar_t>>{ [ks](Stream<wchar_t> x) {
            std::wstring x_str;

            while (!x.IsEmpty())
            {
                x_str.push_back(x.Get());

                x = x.PoppedFront();
            }

            return std::ranges::any_of(ks, std::bind_front(std::equal_to<>{}, x_str))
                ? Result(L""s)
                : Result(x_str);
        } })
    );
}

Parser<int> Eval()
{
    return Bind(Nat(), ParserLambda<int, int>{ [](int x) {
        return Bind(
            Ops(Stream<std::tuple<Parser<wchar_t>, std::function<int(int, int)>>>{
                std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'+'), std::plus<int>{} },
                std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'-'), std::minus<int>{} }
            }), 
            ParserLambda<int, std::function<int(int, int)>>{ [x](std::function<int(int, int)> f) {
                return Bind(Nat(), ParserLambda<int, int>{ [x, f](int y) {
                    return Result(f(x, y));
                } });
            } }
        );
    } });
}

void Eval(std::wstring_view inp)
{
    Show(First(Expr())(inp));
}

class ApplicationType;
class LambdaType;
class LetType;
class VariableType;

class Expression
{
public:
    Expression() = default;

    template <typename T>
    Expression(T t)
        : v{ t }
    {

    }

    std::variant<
        std::monostate,
        std::shared_ptr<ApplicationType>,
        std::shared_ptr<LambdaType>,
        std::shared_ptr<LetType>,
        std::shared_ptr<VariableType>
    > v;
};

class ApplicationType
{
public:
    Expression e1;
    Expression e2;
};

class LambdaType
{
public:
    std::wstring x;
    Expression e;
};

class LetType
{
public:
    std::wstring x;
    Expression e1;
    Expression e2;
};

class VariableType
{
public:
    std::wstring x;
};

Parser<Expression> Atom(int so);
Parser<Expression> Lam(int so);
Parser<Expression> Local(int so);
Parser<Expression> Var();
Parser<Expression> Paren(int so);
Parser<std::wstring> Variable();
Parser<Expression> Expr2(int so);

Parser<std::function<Expression(Expression, Expression)>> Apps(int so)
{
    return Ops(Stream<std::tuple<Parser<Expression>, std::function<Expression(Expression, Expression)>>>{
        std::tuple<Parser<Expression>, std::function<Expression(Expression, Expression)>>{ Expr2(so), [](Expression e1, Expression e2) {
            return std::make_shared<ApplicationType>(e1, e2);
        } } 
    });
}

Parser<Expression> Expr2(int so)
{
    if (so <= 0)
    {
        return Zero<Expression>();
    }

    return Chainl1(Atom(so - 1), Apps(so - 1));
}

Parser<Expression> Atom(int so)
{
    return PlusPlusPlus(
        PlusPlusPlus(
            PlusPlusPlus(Lam(so - 1), Local(so - 1)),
            Var()
        ),
        Paren(so - 1)
    );
}

Parser<Expression> Lam(int so)
{
    return Bind(Symbol(L"\\"), ParserLambda<Expression, std::wstring>{ [so](std::wstring) {
        return Bind(Variable(), ParserLambda<Expression, std::wstring>{ [so](std::wstring x) {
            return Bind(Symbol(L"->"), ParserLambda<Expression, std::wstring>{ [x, so](std::wstring) {
                return Bind(Expr2(so), ParserLambda<Expression, Expression>{ [x](Expression e) {
                    return Result<Expression>(std::make_shared<LambdaType>(x, e));
                    } });
                } });
            } });
        } });
}

Parser<Expression> Local(int so)
{
    return Bind(Symbol(L"let"), ParserLambda<Expression, std::wstring>{ [so](std::wstring) {
        return Bind(Variable(), ParserLambda<Expression, std::wstring>{ [so](std::wstring x) {
            return Bind(Symbol(L"="), ParserLambda<Expression, std::wstring>{ [x, so](std::wstring) {
                return Bind(Expr2(so), ParserLambda<Expression, Expression>{ [x, so](Expression e) {
                    return Bind(Symbol(L"in"), ParserLambda<Expression, std::wstring>{ [x, e, so](std::wstring) {
                        return Bind(Expr2(so), ParserLambda<Expression, Expression>{ [x, e](Expression e2) {
                            return Result<Expression>(std::make_shared<LetType>(x, e, e2));
                            } });
                        } });
                    } });
                } });
            } });
        } });
}

Parser<Expression> Var()
{
    return Bind(Variable(), ParserLambda<Expression, std::wstring>{ [](std::wstring x) {
        return Result<Expression>(std::make_shared<VariableType>(x));
    } });
}

Parser<Expression> Paren(int so)
{
    return Bracket(Symbol(L"("), Expr2(so), Symbol(L")"));
}

Parser<std::wstring> Variable()
{
    return Identifier({ L"let"s, L"in"s });
}

template <>
struct std::formatter<Expression, wchar_t>
{
    template <typename FormatContext>
    auto format(Expression const& v, FormatContext& ctx) const
    {
        auto out{ ctx.out() };

        struct Visitor
        {
            void operator()(std::monostate const& mono)
            {
                out = std::format_to(out, L"Mono");
            }

            void operator()(std::shared_ptr<ApplicationType> const& app)
            {
                out = std::format_to(out, L"App({},{})", app->e1, app->e2);
            }

            void operator()(std::shared_ptr<LambdaType> const& lam)
            {
                out = std::format_to(out, L"Lam({},{})", lam->x, lam->e);
            }

            void operator()(std::shared_ptr<LetType> const& let)
            {
                out = std::format_to(out, L"Let({},{},{})", let->x, let->e1, let->e2);
            }

            void operator()(std::shared_ptr<VariableType> const& var)
            {
                out = std::format_to(out, L"Var({})", var->x);
            }

            decltype(out) out;
        };

        std::visit(Visitor{ out }, v.v);

        return out;
    }

    constexpr auto parse(std::wformat_parse_context& ctx) -> decltype(std::begin(ctx))
    {
        auto it{ std::begin(ctx) };
        auto e_it{ std::end(ctx) };

        if (it != e_it and *it != L'}')
        {
            throw std::format_error{ "ivalid format" };
        }

        return it;
    }
};

int main()
{
    //Show(Word()(L"Yes!"));
    //Show(String(L"hello")(L"hello there"));
    //Show(String(L"hello")(L"helicopter"));
    //Show(Many(Digit())(L"1234a567"));
    //Show(Many(Char(L' '))(L"    Trim"));
    //Show(Nat()(L"12345a"));
    //Show(Int()(L"-12345a"));
    //Show(Ints()(L"[1,2,3,4,5]"));
    //Show(Expr()(L"1+2-(3+4)"));
    //Show(Eval()(L"1+2-(3+4)"));
    //Show(Number()(L"hello"));
    //Show(Number()(L"123"));
    //Show(Colour()(L"yellook"));
    //Show(Identifier({ L"auto"s, L"int"s })(L"int"));
    //Show(Identifier({ L"auto"s, L"int"s })(L"atou"));
    //Show(Identifier({ L"auto"s, L"int"s })(L"auto"));
    //
    //Eval(L"(5+(3-1))+((7-2)^10+(4+2))-(8-3)^2");

    //Show(Local(10)(L"let a = 2 in a + 2"));

    Show(Expr2(10)(L"\\x -> let a = test in hahaha"));

    return 0;
}
