/*
 * @file functional.hpp
 *
 * @copyright Copyright (C) 2025 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef CANARD_FUNCTIONAL_HPP
#define CANARD_FUNCTIONAL_HPP


#include "cuda/kernels/definitions.hpp"

namespace details {

using index_t = int;
using long_index_t = long int;

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    __host__ __device__ constexpr operator value_type() const noexcept { return value; }
    __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
};

template <typename>
struct is_valid_sequence_map;

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t N>
using LongNumber = integral_constant<long_index_t, N>;

template <index_t... Is>
struct Sequence
{
    using Type      = Sequence;
    using data_type = index_t;

    static constexpr index_t mSize = sizeof...(Is);

    __host__ __device__ static constexpr auto Size() { return Number<mSize>{}; }

    __host__ __device__ static constexpr auto GetSize() { return Size(); }

    __host__ __device__ static constexpr index_t At(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[mSize + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    __host__ __device__ static constexpr auto At(Number<I>)
    {
        static_assert(I < mSize, "wrong! I too large");

        return Number<At(I)>{};
    }

    template <index_t I>
    __host__ __device__ static constexpr auto Get(Number<I>)
    {
        return At(Number<I>{});
    }

    template <typename I>
    __host__ __device__ constexpr auto operator[](I i) const
    {
        return At(i);
    }

    // template <index_t... IRs>
    // __host__ __device__ static constexpr auto ReorderGivenNew2Old(Sequence<IRs...> /*new2old*/)
    // {
    //     static_assert(sizeof...(Is) == sizeof...(IRs),
    //                   "wrong! reorder map should have the same size as Sequence to be rerodered");

    //     // static_assert(is_valid_sequence_map<Sequence<IRs...>>::value, "wrong! invalid reorder map");

    //     return Sequence<Type::At(Number<IRs>{})...>{};
    // }

    // // MapOld2New is Sequence<...>
    // template <typename MapOld2New>
    // __host__ __device__ static constexpr auto ReorderGivenOld2New(MapOld2New)
    // {
    //     static_assert(MapOld2New::Size() == Size(),
    //                   "wrong! reorder map should have the same size as Sequence to be rerodered");

    //     static_assert(is_valid_sequence_map<MapOld2New>::value, "wrong! invalid reorder map");

    //     return ReorderGivenNew2Old(typename sequence_map_inverse<MapOld2New>::type{});
    // }

    // __host__ __device__ static constexpr auto Reverse()
    // {
    //     return typename sequence_reverse<Type>::type{};
    // }

    __host__ __device__ static constexpr auto Front()
    {
        static_assert(mSize > 0, "wrong!");
        return At(Number<0>{});
    }

    __host__ __device__ static constexpr auto Back()
    {
        static_assert(mSize > 0, "wrong!");
        return At(Number<mSize - 1>{});
    }

    __host__ __device__ static constexpr auto PopFront() { return sequence_pop_front(Type{}); }

    __host__ __device__ static constexpr auto PopBack() { return sequence_pop_back(Type{}); }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Sequence<Xs...>)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushFront(Number<Xs>...)
    {
        return Sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Sequence<Xs...>)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Xs>
    __host__ __device__ static constexpr auto PushBack(Number<Xs>...)
    {
        return Sequence<Is..., Xs...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Number<Ns>...)
    {
        return Sequence<Type::At(Number<Ns>{})...>{};
    }

    template <index_t... Ns>
    __host__ __device__ static constexpr auto Extract(Sequence<Ns...>)
    {
        return Sequence<Type::At(Number<Ns>{})...>{};
    }

    // template <index_t I, index_t X>
    // __host__ __device__ static constexpr auto Modify(Number<I>, Number<X>)
    // {
    //     static_assert(I < Size(), "wrong!");

    //     using seq_split          = sequence_split<Type, I>;
    //     constexpr auto seq_left  = typename seq_split::left_type{};
    //     constexpr auto seq_right = typename seq_split::right_type{}.PopFront();

    //     return seq_left.PushBack(Number<X>{}).PushBack(seq_right);
    // }

    template <typename F>
    __host__ __device__ static constexpr auto Transform(F f)
    {
        return Sequence<f(Is)...>{};
    }

    // __host__ __device__ static void Print()
    // {
    //     printf("{");
    //     printf("size %d, ", index_t{Size()});
    //     static_for<0, Size(), 1>{}([&](auto i) { printf("%d ", At(i).value); });
    //     printf("}");
    // }
};


template <bool predicate, class X, class Y>
struct conditional;

template <class X, class Y>
struct conditional<true, X, Y>
{
    using type = X;
};

template <class X, class Y>
struct conditional<false, X, Y>
{
    using type = Y;
};

// merge sequence
template <typename Seq, typename... Seqs>
struct sequence_merge
{
    using type = typename sequence_merge<Seq, typename sequence_merge<Seqs...>::type>::type;
};

template <index_t... Xs, index_t... Ys>
struct sequence_merge<Sequence<Xs...>, Sequence<Ys...>>
{
    using type = Sequence<Xs..., Ys...>;
};

template <typename Seq>
struct sequence_merge<Seq>
{
    using type = Seq;
};

// generate sequence
template <index_t NSize, typename F>
struct sequence_gen
{
    template <index_t IBegin, index_t NRemain, typename G>
    struct sequence_gen_impl
    {
        static constexpr index_t NRemainLeft  = NRemain / 2;
        static constexpr index_t NRemainRight = NRemain - NRemainLeft;
        static constexpr index_t IMiddle      = IBegin + NRemainLeft;

        using type = typename sequence_merge<
            typename sequence_gen_impl<IBegin, NRemainLeft, G>::type,
            typename sequence_gen_impl<IMiddle, NRemainRight, G>::type>::type;
    };

    template <index_t I, typename G>
    struct sequence_gen_impl<I, 1, G>
    {
        static constexpr index_t Is = G{}(Number<I>{});
        using type                  = Sequence<Is>;
    };

    template <index_t I, typename G>
    struct sequence_gen_impl<I, 0, G>
    {
        using type = Sequence<>;
    };

    using type = typename sequence_gen_impl<0, NSize, F>::type;
};


template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    struct F
    {
        __host__ __device__ constexpr index_t operator()(index_t i) const
        {
            return i * Increment + IBegin;
        }
    };

    using type0 = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
    using type1 = Sequence<>;

    static constexpr bool kHasContent =
        (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

    using type = typename conditional<kHasContent, type0, type1>::type;
};

// template <typename Values, typename Ids, typename Compare>
// struct sequence_sort_impl
// {
//     template <typename LeftValues,
//               typename LeftIds,
//               typename RightValues,
//               typename RightIds,
//               typename MergedValues,
//               typename MergedIds,
//               typename Comp>
//     struct sorted_sequence_merge_impl
//     {
//         static constexpr bool choose_left = LeftValues::front() < RightValues::front();

//         static constexpr index_t chosen_value =
//             choose_left ? LeftValues::front() : RightValues::front();
//         static constexpr index_t chosen_id = choose_left ? LeftIds::front() : RightIds::front();

//         using new_merged_values = decltype(MergedValues::push_back(number<chosen_value>{}));
//         using new_merged_ids    = decltype(MergedIds::push_back(number<chosen_id>{}));

//         using new_left_values = typename std::
//             conditional<choose_left, decltype(LeftValues::pop_front()), LeftValues>::type;
//         using new_left_ids =
//             typename std::conditional<choose_left, decltype(LeftIds::pop_front()), LeftIds>::type;

//         using new_right_values = typename std::
//             conditional<choose_left, RightValues, decltype(RightValues::pop_front())>::type;
//         using new_right_ids =
//             typename std::conditional<choose_left, RightIds, decltype(RightIds::pop_front())>::type;

//         using merge = sorted_sequence_merge_impl<new_left_values,
//                                                  new_left_ids,
//                                                  new_right_values,
//                                                  new_right_ids,
//                                                  new_merged_values,
//                                                  new_merged_ids,
//                                                  Comp>;
//         // this is output
//         using merged_values = typename merge::merged_values;
//         using merged_ids    = typename merge::merged_ids;
//     };

//     template <typename LeftValues,
//               typename LeftIds,
//               typename MergedValues,
//               typename MergedIds,
//               typename Comp>
//     struct sorted_sequence_merge_impl<LeftValues,
//                                       LeftIds,
//                                       sequence<>,
//                                       sequence<>,
//                                       MergedValues,
//                                       MergedIds,
//                                       Comp>
//     {
//         using merged_values = typename sequence_merge<MergedValues, LeftValues>::type;
//         using merged_ids    = typename sequence_merge<MergedIds, LeftIds>::type;
//     };

//     template <typename RightValues,
//               typename RightIds,
//               typename MergedValues,
//               typename MergedIds,
//               typename Comp>
//     struct sorted_sequence_merge_impl<sequence<>,
//                                       sequence<>,
//                                       RightValues,
//                                       RightIds,
//                                       MergedValues,
//                                       MergedIds,
//                                       Comp>
//     {
//         using merged_values = typename sequence_merge<MergedValues, RightValues>::type;
//         using merged_ids    = typename sequence_merge<MergedIds, RightIds>::type;
//     };

//     template <typename LeftValues,
//               typename LeftIds,
//               typename RightValues,
//               typename RightIds,
//               typename Comp>
//     struct sorted_sequence_merge
//     {
//         using merge = sorted_sequence_merge_impl<LeftValues,
//                                                  LeftIds,
//                                                  RightValues,
//                                                  RightIds,
//                                                  sequence<>,
//                                                  sequence<>,
//                                                  Comp>;

//         using merged_values = typename merge::merged_values;
//         using merged_ids    = typename merge::merged_ids;
//     };

//     static constexpr index_t nsize = Values::size();

//     using split_unsorted_values = sequence_split<Values, nsize / 2>;
//     using split_unsorted_ids    = sequence_split<Ids, nsize / 2>;

//     using left_unsorted_values = typename split_unsorted_values::left_type;
//     using left_unsorted_ids    = typename split_unsorted_ids::left_type;
//     using left_sort          = sequence_sort_impl<left_unsorted_values, left_unsorted_ids, Compare>;
//     using left_sorted_values = typename left_sort::sorted_values;
//     using left_sorted_ids    = typename left_sort::sorted_ids;

//     using right_unsorted_values = typename split_unsorted_values::right_type;
//     using right_unsorted_ids    = typename split_unsorted_ids::right_type;
//     using right_sort = sequence_sort_impl<right_unsorted_values, right_unsorted_ids, Compare>;
//     using right_sorted_values = typename right_sort::sorted_values;
//     using right_sorted_ids    = typename right_sort::sorted_ids;

//     using merged_sorted = sorted_sequence_merge<left_sorted_values,
//                                                 left_sorted_ids,
//                                                 right_sorted_values,
//                                                 right_sorted_ids,
//                                                 Compare>;

//     using sorted_values = typename merged_sorted::merged_values;
//     using sorted_ids    = typename merged_sorted::merged_ids;
// };


// template <index_t ValueX, index_t ValueY, index_t IdX, index_t IdY, typename Compare>
// struct sequence_sort_impl<sequence<ValueX, ValueY>, sequence<IdX, IdY>, Compare>
// {
//     static constexpr bool choose_x = Compare{}(ValueX, ValueY);

//     using sorted_values = typename std::
//         conditional<choose_x, sequence<ValueX, ValueY>, sequence<ValueY, ValueX>>::type;
//     using sorted_ids =
//         typename std::conditional<choose_x, sequence<IdX, IdY>, sequence<IdY, IdX>>::type;
// };

// template <index_t Value, index_t Id, typename Compare>
// struct sequence_sort_impl<sequence<Value>, sequence<Id>, Compare>
// {
//     using sorted_values = sequence<Value>;
//     using sorted_ids    = sequence<Id>;
// };

// template <typename Compare>
// struct sequence_sort_impl<sequence<>, sequence<>, Compare>
// {
//     using sorted_values = sequence<>;
//     using sorted_ids    = sequence<>;
// };

// template <typename Values, typename Compare>
// struct sequence_sort
// {
//     using unsorted_ids = typename arithmetic_sequence_gen<0, Values::size(), 1>::type;
//     using sort         = sequence_sort_impl<Values, unsorted_ids, Compare>;

//     // this is output
//     using type                = typename sort::sorted_values;
//     using sorted2unsorted_map = typename sort::sorted_ids;
// };


// template <index_t IEnd>
// struct arithmetic_sequence_gen<0, IEnd, 1>
// {
//     template <typename T, T... Ints>
//     struct WrapSequence
//     {
//         using type = Sequence<Ints...>;
//     };
//     // https://reviews.llvm.org/D13786
//     using type = typename __make_integer_seq<WrapSequence, index_t, IEnd>::type;
// };


struct swallow
{
    template <typename... Ts>
    __host__ __device__ constexpr swallow(Ts&&...)
    {
    }
};

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<Sequence<Is...>>
{
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        swallow{(f(Number<Is>{}), 0)...};
    }
};

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    __host__ __device__ constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");
    }

    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(
            f);
    }
};

} // namespace details

#endif
