/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "pp_3wide.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

constexpr IndexType make_pawn_id(Color color, Square square) {
    assert(square >= SQ_A2 && square <= SQ_H7);
    return 48 * int(color) + square - SQ_A2;
}

#ifdef USE_AVX512ICL
static inline __m256i pp_idx_epi16(__m256i a, __m256i b) {
    const __m256i hi   = _mm256_max_epu16(a, b);
    const __m256i lo   = _mm256_min_epu16(a, b);
    const __m256i prod = _mm256_mullo_epi16(hi, _mm256_sub_epi16(hi, _mm256_set1_epi16(1)));
    return _mm256_add_epi16(_mm256_add_epi16(_mm256_srli_epi16(prod, 1), lo),
                            _mm256_set1_epi16(i16(PP_3Wide::IndexBase)));
}
#endif

sf_always_inline IndexType PP_3Wide::make_index(
  Color perspective, Color color, Square from, Square to, Color pairedColor, Square ksq) {
    const i8 orientation   = OrientTBL[ksq] ^ (56 * perspective);
    unsigned from_oriented = u8(from) ^ orientation;
    unsigned to_oriented   = u8(to) ^ orientation;

    Color color_oriented       = Color(color ^ perspective);
    Color pairedColor_oriented = Color(pairedColor ^ perspective);

    assert(from_oriented >= SQ_A2 && from_oriented <= SQ_H7);
    assert(to_oriented >= SQ_A2 && to_oriented <= SQ_H7);

    const IndexType idA = make_pawn_id(color_oriented, Square(from_oriented));
    const IndexType idB = make_pawn_id(pairedColor_oriented, Square(to_oriented));
    const IndexType hi  = std::max(idA, idB);
    const IndexType lo  = std::min(idA, idB);

    return hi * (hi - 1) / 2 + lo + IndexBase;
}

void PP_3Wide::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq      = pos.square<KING>(perspective);
    const u32      orient   = OrientTBL[ksq] ^ (56 * perspective);
    const Bitboard friendly = pos.pieces(perspective, PAWN);
    const Bitboard enemy    = pos.pieces(~perspective, PAWN);

    Bitboard bb = enemy;
    while (bb)
    {
        const Square   from = pop_lsb(bb);
        const Bitboard band = pawn_pair_bb(from);
        if (!(band & (bb | friendly)))
            continue;
        const u32 idA   = (u32(from) ^ orient) + 40;
        const u32 baseA = IndexBase + idA * (idA - 1) / 2;
        for (Bitboard ee = band & bb; ee;)
        {
            const u32 idB = (u32(pop_lsb(ee)) ^ orient) + 40;
            active.push_back(idA > idB ? baseA + idB : IndexBase + idB * (idB - 1) / 2 + idA);
        }
        const u32 baseA_adj = baseA - 8;
        for (Bitboard ef = band & friendly; ef;)
            active.push_back(baseA_adj + (u32(pop_lsb(ef)) ^ orient));
    }

    bb = friendly;
    while (bb)
    {
        const Square   from = pop_lsb(bb);
        const Bitboard ffk  = pawn_pair_bb(from) & bb;
        if (!ffk)
            continue;
        const u32 idA   = (u32(from) ^ orient) - 8;
        const u32 baseA = IndexBase + idA * (idA - 1) / 2;
        for (Bitboard f = ffk; f;)
        {
            const u32 idB = (u32(pop_lsb(f)) ^ orient) - 8;
            active.push_back(idA > idB ? baseA + idB : IndexBase + idB * (idB - 1) / 2 + idA);
        }
    }
}

void PP_3Wide::append_changed_indices(Color                                    perspective,
                                      Square                                   ksq,
                                      const DiffType&                          diff,
                                      IndexList& RESTRICT                      removed,
                                      IndexList& RESTRICT                      added,
                                      [[maybe_unused]] const ThreatWeightType* prefetchBase) {

    const Bitboard whiteBefore = diff.before[WHITE];
    const Bitboard blackBefore = diff.before[BLACK];
    const Bitboard whiteAfter  = diff.after[WHITE];
    const Bitboard blackAfter  = diff.after[BLACK];

    if (whiteBefore == whiteAfter && blackBefore == blackAfter)
        return;

    const Bitboard unchanged = (whiteBefore & whiteAfter) | (blackBefore & blackAfter);

#ifdef USE_AVX512ICL
    const u8      orientation = OrientTBL[ksq] ^ u8(56 * perspective);
    const __m512i iota        = AllSquares;
    const __m512i adjusted =
      _mm512_sub_epi8(_mm512_xor_si512(iota, _mm512_set1_epi8(orientation)), _mm512_set1_epi8(8));

    auto generate = [&](Bitboard pawnsW, Bitboard pawnsB,
                        IndexList& RESTRICT out) __attribute__((always_inline)) {
        Bitboard u = (pawnsW | pawnsB) & ~unchanged;
        if (!u)
            return;

        const Bitboard friendly = perspective == WHITE ? pawnsW : pawnsB;
        const Bitboard enemy    = perspective == WHITE ? pawnsB : pawnsW;
        const __m512i  ids      = _mm512_mask_blend_epi8(
          friendly, _mm512_add_epi8(adjusted, _mm512_set1_epi8(48)), adjusted);

        for (; u;)
        {
            const Square   a        = pop_lsb(u);
            const Bitboard partners = pawn_pair_bb(a) & (unchanged | u);
            const int      n        = popcount(partners);
            if (!n)
                continue;

            const u16     colorOff = (enemy & a) ? 48 : 0;
            const u16     aId      = u16(((u8(a) ^ orientation) - 8) + colorOff);
            const __m256i pids     = _mm256_cvtepu8_epi16(
              _mm512_castsi512_si128(_mm512_maskz_compress_epi8(partners, ids)));
            const __m256i feats = pp_idx_epi16(_mm256_set1_epi16(aId), pids);

            u16* w = out.make_space(n);
            _mm256_storeu_epi16(w, feats);
        }
    };
#else
    const u32 orient = OrientTBL[ksq] ^ (56 * perspective);

    auto generate = [&](Bitboard pawnsW, Bitboard pawnsB,
                        IndexList& RESTRICT out) __attribute__((always_inline)) {
        const Bitboard pawnsF = perspective == WHITE ? pawnsW : pawnsB;
        const Bitboard pawnsE = perspective == WHITE ? pawnsB : pawnsW;

        auto push = [&](IndexType index) __attribute__((always_inline)) {
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(prefetchBase + uintptr_t(index) * L1);
            out.push_back(index);
        };
        for (Bitboard u = (pawnsW | pawnsB) & ~unchanged; u;)
        {
            const Square   a    = pop_lsb(u);
            const Bitboard mask = pawn_pair_bb(a) & (unchanged | u);
            if (!mask)
                continue;
            if (pawnsE & a)
            {
                const u32 idA   = (u32(a) ^ orient) + 40;
                const u32 baseA = IndexBase + idA * (idA - 1) / 2;
                for (Bitboard pe = pawnsE & mask; pe;)
                {
                    const u32 idB = (u32(pop_lsb(pe)) ^ orient) + 40;
                    push(idA > idB ? baseA + idB : IndexBase + idB * (idB - 1) / 2 + idA);
                }
                const u32 baseA_adj = baseA - 8;
                for (Bitboard pf = pawnsF & mask; pf;)
                    push(baseA_adj + (u32(pop_lsb(pf)) ^ orient));
            }
            else
            {
                const u32 idA   = (u32(a) ^ orient) - 8;
                const u32 baseA = IndexBase + idA * (idA - 1) / 2;
                for (Bitboard pe = pawnsE & mask; pe;)
                {
                    const u32 idB = (u32(pop_lsb(pe)) ^ orient) + 40;
                    push(IndexBase + idB * (idB - 1) / 2 + idA);
                }
                for (Bitboard pf = pawnsF & mask; pf;)
                {
                    const u32 idB = (u32(pop_lsb(pf)) ^ orient) - 8;
                    push(idA > idB ? baseA + idB : IndexBase + idB * (idB - 1) / 2 + idA);
                }
            }
        }
    };
#endif

    generate(whiteAfter, blackAfter, added);
    generate(whiteBefore, blackBefore, removed);
}

void PP_3Wide::append_changed_indices_both(Square                                   white_ksq,
                                           Square                                   black_ksq,
                                           const DiffType&                          diff,
                                           IndexList& RESTRICT                      white_removed,
                                           IndexList& RESTRICT                      white_added,
                                           IndexList& RESTRICT                      black_removed,
                                           IndexList& RESTRICT                      black_added,
                                           [[maybe_unused]] const ThreatWeightType* prefetchBase) {

    const Bitboard whiteBefore = diff.before[WHITE];
    const Bitboard blackBefore = diff.before[BLACK];
    const Bitboard whiteAfter  = diff.after[WHITE];
    const Bitboard blackAfter  = diff.after[BLACK];

    if (whiteBefore == whiteAfter && blackBefore == blackAfter)
        return;

    const Bitboard unchanged = (whiteBefore & whiteAfter) | (blackBefore & blackAfter);

#ifdef USE_AVX512ICL
    const u8      wOrient = OrientTBL[white_ksq];
    const u8      bOrient = OrientTBL[black_ksq] ^ 56;
    const __m512i iota    = AllSquares;
    const __m512i wAdj =
      _mm512_sub_epi8(_mm512_xor_si512(iota, _mm512_set1_epi8(wOrient)), _mm512_set1_epi8(8));
    const __m512i bAdj =
      _mm512_sub_epi8(_mm512_xor_si512(iota, _mm512_set1_epi8(bOrient)), _mm512_set1_epi8(8));

    auto generate = [&](Bitboard pawnsW, Bitboard pawnsB, IndexList& RESTRICT white_out,
                        IndexList& RESTRICT black_out) __attribute__((always_inline)) {
        Bitboard u = (pawnsW | pawnsB) & ~unchanged;
        if (!u)
            return;

        const __m512i wIds =
          _mm512_mask_blend_epi8(pawnsW, _mm512_add_epi8(wAdj, _mm512_set1_epi8(48)), wAdj);
        const __m512i bIds =
          _mm512_mask_blend_epi8(pawnsB, _mm512_add_epi8(bAdj, _mm512_set1_epi8(48)), bAdj);

        for (; u;)
        {
            const Square   a        = pop_lsb(u);
            const Bitboard partners = pawn_pair_bb(a) & (unchanged | u);
            const int      n        = popcount(partners);
            if (!n)
                continue;

            const u16     wColorOff = (pawnsB & a) ? 48 : 0;
            const u16     bColorOff = wColorOff ^ 48;
            const u16     wAId      = u16(((u8(a) ^ wOrient) - 8) + wColorOff);
            const u16     bAId      = u16(((u8(a) ^ bOrient) - 8) + bColorOff);
            const __m256i wPids     = _mm256_cvtepu8_epi16(
              _mm512_castsi512_si128(_mm512_maskz_compress_epi8(partners, wIds)));
            const __m256i bPids     = _mm256_cvtepu8_epi16(
              _mm512_castsi512_si128(_mm512_maskz_compress_epi8(partners, bIds)));

            _mm256_storeu_epi16(white_out.make_space(n),
                                pp_idx_epi16(_mm256_set1_epi16(wAId), wPids));
            _mm256_storeu_epi16(black_out.make_space(n),
                                pp_idx_epi16(_mm256_set1_epi16(bAId), bPids));
        }
    };
#else
    const u32 w_orient = OrientTBL[white_ksq];
    const u32 b_orient = OrientTBL[black_ksq] ^ 56;

    auto generate = [&](Bitboard pawnsW, Bitboard pawnsB, IndexList& RESTRICT white_out,
                        IndexList& RESTRICT black_out) __attribute__((always_inline)) {
        auto push_pair = [&](IndexType white_index,
                             IndexType black_index) __attribute__((always_inline)) {
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(prefetchBase
                                                         + uintptr_t(white_index) * L1);
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(prefetchBase
                                                         + uintptr_t(black_index) * L1);

            white_out.push_back(white_index);
            black_out.push_back(black_index);
        };

        for (Bitboard updated = (pawnsW | pawnsB) & ~unchanged; updated;)
        {
            const Square   from = pop_lsb(updated);
            const Bitboard mask = pawn_pair_bb(from) & (unchanged | updated);
            if (!mask)
                continue;

            if (pawnsB & from)
            {
                const u32 w_idA   = (u32(from) ^ w_orient) + 40;
                const u32 b_idA   = (u32(from) ^ b_orient) - 8;
                const u32 w_baseA = IndexBase + w_idA * (w_idA - 1) / 2;
                const u32 b_baseA = IndexBase + b_idA * (b_idA - 1) / 2;

                for (Bitboard paired = pawnsB & mask; paired;)
                {
                    const Square    to    = pop_lsb(paired);
                    const u32       w_idB = (u32(to) ^ w_orient) + 40;
                    const u32       b_idB = (u32(to) ^ b_orient) - 8;
                    const IndexType white_index =
                      w_idA > w_idB ? w_baseA + w_idB : IndexBase + w_idB * (w_idB - 1) / 2 + w_idA;
                    const IndexType black_index =
                      b_idA > b_idB ? b_baseA + b_idB : IndexBase + b_idB * (b_idB - 1) / 2 + b_idA;
                    push_pair(white_index, black_index);
                }
                const u32 w_baseA_adj = w_baseA - 8;
                for (Bitboard paired = pawnsW & mask; paired;)
                {
                    const Square    to          = pop_lsb(paired);
                    const u32       b_idB       = (u32(to) ^ b_orient) + 40;
                    const IndexType white_index = w_baseA_adj + (u32(to) ^ w_orient);
                    const IndexType black_index = IndexBase + b_idB * (b_idB - 1) / 2 + b_idA;
                    push_pair(white_index, black_index);
                }
            }
            else
            {
                const u32 w_idA   = (u32(from) ^ w_orient) - 8;
                const u32 b_idA   = (u32(from) ^ b_orient) + 40;
                const u32 w_baseA = IndexBase + w_idA * (w_idA - 1) / 2;
                const u32 b_baseA = IndexBase + b_idA * (b_idA - 1) / 2;

                const u32 b_baseA_adj = b_baseA - 8;
                for (Bitboard paired = pawnsB & mask; paired;)
                {
                    const Square    to          = pop_lsb(paired);
                    const u32       w_idB       = (u32(to) ^ w_orient) + 40;
                    const IndexType white_index = IndexBase + w_idB * (w_idB - 1) / 2 + w_idA;
                    const IndexType black_index = b_baseA_adj + (u32(to) ^ b_orient);
                    push_pair(white_index, black_index);
                }
                for (Bitboard paired = pawnsW & mask; paired;)
                {
                    const Square    to    = pop_lsb(paired);
                    const u32       w_idB = (u32(to) ^ w_orient) - 8;
                    const u32       b_idB = (u32(to) ^ b_orient) + 40;
                    const IndexType white_index =
                      w_idA > w_idB ? w_baseA + w_idB : IndexBase + w_idB * (w_idB - 1) / 2 + w_idA;
                    const IndexType black_index =
                      b_idA > b_idB ? b_baseA + b_idB : IndexBase + b_idB * (b_idB - 1) / 2 + b_idA;
                    push_pair(white_index, black_index);
                }
            }
        }
    };
#endif

    generate(whiteAfter, blackAfter, white_added, black_added);
    generate(whiteBefore, blackBefore, white_removed, black_removed);
}

}  // namespace Stockfish::Eval::NNUE::Features
