import Parser;
import std;

using namespace StreamModule;
using namespace ParserModule;
using namespace std::literals;

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

template <typename... Ts>
struct std::formatter<std::tuple<Ts...>, wchar_t>
{
    template <typename FormatContext, std::size_t... Is>
    void PrintTuple(FormatContext& ctx, std::tuple<Ts...> const& t, std::index_sequence<Is...>) const
    {
        auto out{ ctx.out() };
        
        *out++ = L'(';
        ((Is == 0 ? out = std::format_to(out, L"{}", std::get<Is>(t)) : out = std::format_to(out, L",{}", std::get<Is>(t))), ...);
        *out++ = L')';
    }

    template <typename FormatContext>
    auto format(std::tuple<Ts...> const& t, FormatContext& ctx) const
    {
        PrintTuple(ctx, t, std::make_index_sequence<sizeof...(Ts)>{});

        return ctx.out();
    }

    constexpr auto parse(std::wformat_parse_context& ctx) -> decltype(std::begin(ctx))
    {
        auto it{ std::begin(ctx) };
        auto e_it{ std::end(ctx) };

        if (it != e_it and *it != L'}')
        {
            throw std::format_error{ "invalid format" };
        }

        return it;
    }
};

Stream<std::tuple<int, int, int>> Triples()
{
    return IntsFrom(1) >>= [](int z) {
        return Ints(1, z) >>= [z](int x) {
            return Ints(x, z) >>= [x, z](int y) {
                return Mthen(Guard(x * x + y * y == z * z), [x, y, z] {
                    return Mreturn(std::tuple{ x, y, z });
                });
            };
        };
    };
}

auto Item()
{
    return Parser<wchar_t>::Make([](std::wstring_view inp) {
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
        return pa(inp) >>= [pb](std::tuple<A, std::wstring_view> t1) {
            auto const& [v, inp1] { t1 };

            return pb(inp1) >>= [](std::tuple<B, std::wstring_view> t2) {
                auto const& [w, inp2] { t2 };

                return Mreturn(std::tuple{ std::tuple{ v, w }, inp2 });
            };
        };
    });
}

auto Sat(std::function<bool(wchar_t)> p)
{
    return Item() >>= [p](wchar_t x) {
        return p(x) ? Result(x) : Zero<wchar_t>();
    };
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
    auto ne_word{ Letter() >>= [](wchar_t x) {
        return Word() >>= [x](std::wstring xs) {
            return Result(std::wstring{ x } + xs);
        };
    } };

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

    return Char(x) >>= [x, xs](wchar_t x) {
        return String(xs) >>= [x, xs](std::wstring) {
            return Result(std::wstring{ x } + xs);
        };
    };
}

template <typename A>
Parser<Stream<A>> Many(Parser<A> p);

//template <typename A>
//Parser<Stream<A>> Many(Parser<A> p)
//{
//    auto ne_many{ p >>= [p](A x) {
//        return Many(p) >>= [x](Stream<A> xs) {
//            return Result(Stream<A>{ x, xs });
//        };
//    } };
//
//    return Plus(ne_many, Result(Stream<A>{}));
//}

Parser<Stream<wchar_t>> Ident()
{
    return Lower() >>= [](wchar_t x) {
        return Many(AlphaNum()) >>= [x](Stream<wchar_t> xs) {
            return Result(Stream<wchar_t>{ x, xs });
        };
    };
}

template <typename A>
Parser<Stream<A>> Many1(Parser<A> p)
{
    return p >>= [p](A x) {
        return Many(p) >>= [x](Stream<A> xs) {
            return Result(Stream<A>{ x, xs });
        };
    };
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
        Char(L'-') >>= [](wchar_t) {
            return Nat() >>= [](int n) {
                return Result(-n);
                };
            },
        Nat()
    );
}

template <typename A, typename B>
Parser<Stream<A>> SepBy1(Parser<A> p, Parser<B> sep)
{
    return p >>= [p, sep](A x) {
        auto many{
            Many(
                sep >>= [p](B) {
                    return p >>= [](A y) {
                        return Result(y);
                    };
                }
            )
        };

        return many >>= [x](Stream<A> xs) {
            return Result(Stream<A>{ x, xs });
        };
    };
}

template <typename A, typename B, typename C>
Parser<B> Bracket(Parser<A> open, Parser<B> p, Parser<C> close)
{
    return open >>= [p, close](A) {
        return p >>= [close](B x) {
            return close >>= [x](C) {
                return Result(x);
            };
        };
    };
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

            return p >>= [op](A) {
                return Result(op);
            };
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
            op >>= [p, op, x](std::function<A(A, A)> const& f) {
                return p >>= [p, op, x, f](A y) {
                    return Rest(p, op)(f(x, y));
                };
            },
            Result(x)
        );
    };
}

template <typename A>
Parser<A> Chainl1(Parser<A> p, Parser<std::function<A(A, A)>> const& op)
{
    return p >>= Rest(p, op);
}

Parser<int> Nat()
{
    std::function<int(int, int)> op{ [](int m, int n) { return 10 * m + n; } };

    return Chainl1(
        Digit() >>= [](wchar_t x) {
            return Result(x - L'0');
        },
        Result(op)
    );
}

template <typename A>
Parser<A> Chainr1(Parser<A> p, Parser<std::function<A(A, A)>> const& op)
{
    return p >>= [p, op](A x) {
        return Plus(
            op >>= [p, op, x](std::function<A(A, A)> const& f) {
                return Chainr1(p, op) >>= [f, x](A y) {
                    return Result(f(x, y));
                };
            },
            Result(x)
        );
    };
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
        p >>= [p](A x) {
            return Many(p) >>= [x](Stream<A> xs) {
                return Result(Stream<A>{ x, xs });
            };
        },
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

    return Many1(Sat(is_space)) >>= [](Stream<wchar_t>) {
        return Result(std::tuple<>{});
    };
}

Parser<std::tuple<>> Comment()
{
    return String(L"--") >>= [](std::wstring) {
        return Many(Sat([](wchar_t x) { return x != L'\n'; })) >>= [](Stream<wchar_t>) {
            return Result(std::tuple<>{});
        };
    };
}

Parser<std::tuple<>> Junk()
{
    return Many(PlusPlusPlus(Spaces(), Comment())) >>= [](Stream<std::tuple<>>) {
        return Result(std::tuple<>{});
    };
}

template <typename A>
Parser<A> Parse(Parser<A> p)
{
    return Junk() >>= [](std::tuple<>) {
        return p >>= [](A v) {
            return Result(v);
        };
    };
}

template <typename A>
Parser<A> Token(Parser<A> p)
{
    return p >>= [](A v) {
        return Junk() >>= [v](std::tuple<>) {
            return Result(v);
        };
    };
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
        Ident() >>= [ks](Stream<wchar_t> x) {
            std::wstring x_str;

            while (!x.IsEmpty())
            {
                x_str.push_back(x.Get());

                x = x.PoppedFront();
            }

            return std::ranges::any_of(ks, std::bind_front(std::equal_to<>{}, x_str))
                ? Result(L""s)
                : Result(x_str);
        }
    );
}

Parser<int> Eval()
{
    return Nat() >>= [](int x) {
        return Ops(Stream<std::tuple<Parser<wchar_t>, std::function<int(int, int)>>>{
                std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'+'), std::plus<int>{} },
                std::tuple<Parser<wchar_t>, std::function<int(int, int)>>{ Char(L'-'), std::minus<int>{} }
            }) >>= [x](std::function<int(int, int)> f) {
                return Nat() >>= [x, f](int y) {
                    return Result(f(x, y));
                };
            };
        };
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
    return Symbol(L"\\") >>= [so](std::wstring) {
        return Variable() >>= [so](std::wstring x) {
            return Symbol(L"->") >>= [x, so](std::wstring) {
                return Expr2(so) >>= [x](Expression e) {
                    return Result<Expression>(std::make_shared<LambdaType>(x, e));
                };
            };
        };
    };
}

Parser<Expression> Local(int so)
{
    return Symbol(L"let") >>= [so](std::wstring) {
        return Variable() >>= [so](std::wstring x) {
            return Symbol(L"=") >>= [x, so](std::wstring) {
                return Expr2(so) >>= [x, so](Expression e) {
                    return Symbol(L"in") >>= [x, e, so](std::wstring) {
                        return Expr2(so) >>= [x, e](Expression e2) {
                            return Result<Expression>(std::make_shared<LetType>(x, e, e2));
                        };
                    };
                };
            };
        };
    };
}

Parser<Expression> Var()
{
    return Variable() >>= [](std::wstring x) {
        return Result<Expression>(std::make_shared<VariableType>(x));
    };
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
    std::wcout << std::format(L"{}\n", Triples().Take(20));

    Show(Word()(L"Yes!"));
    Show(String(L"hello")(L"hello there"));
    Show(String(L"hello")(L"helicopter"));
    Show(Many(Digit())(L"1234a567"));
    Show(Many(Char(L' '))(L"    Trim"));
    Show(Nat()(L"12345a"));
    Show(Int()(L"-12345a"));
    Show(Ints()(L"[1,2,3,4,5]"));
    Show(Expr()(L"1+2-(3+4)"));
    Show(Eval()(L"1+2-(3+4)"));
    Show(Number()(L"hello"));
    Show(Number()(L"123"));
    Show(Colour()(L"yellook"));
    Show(Identifier({ L"auto"s, L"int"s })(L"int"));
    Show(Identifier({ L"auto"s, L"int"s })(L"atou"));
    Show(Identifier({ L"auto"s, L"int"s })(L"auto"));
    
    Eval(L"(5+(3-1))+((7-2)^10+(4+2))-(8-3)^2");

    Show(Expr2(10)(L"\\x -> let a = test in hahaha"));

    return 0;
}
