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

//Definition of input features FullThreats of NNUE evaluation function

#include "full_threats.h"

#include <array>
#include <cstdint>
#include <initializer_list>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../misc.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

struct alignas(64) ThreatLUTs {
    // [attacked][attacker]: low u16 = base[from >= to], high u16 = base[from < to] - 0x8000u
    std::array<std::array<u32, PIECE_NB>, PIECE_NB> lut1{};
    // [pt][(from << 6) | to]: bit 15 = (from < to), bits 0..10 = piece-pair offset
    std::array<std::array<i16, SQUARE_NB * SQUARE_NB>, PIECE_TYPE_NB> lut2{};
};

constexpr std::array<Piece, 12> AllPieces = {
  W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
};

constexpr auto init_threat_luts() {
    ThreatLUTs                     tables{};
    std::array<u16, PIECE_TYPE_NB> offsets{};

    constexpr u32 ExcludedEntry =
      u32(FullThreats::Dimensions) | (u32(FullThreats::Dimensions - 0x8000u) << 16);
    for (auto& row : tables.lut1)
        for (auto& entry : row)
            entry = ExcludedEntry;

    for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING})
    {
        u16 offset = 0;
        for (Square from = SQ_A1; from <= SQ_H8; ++from)
        {
            for (Square to = SQ_A1; to <= SQ_H8; ++to)
            {
                Bitboard  attacks = (pt != PAWN) ? Attacks::PseudoAttacks[pt][from]
                                  : (to < from)  ? Attacks::PseudoAttacks[BLACK][from]
                                                 : Attacks::PseudoAttacks[WHITE][from];
                const u16 idx     = offset + constexpr_popcount(((1ULL << to) - 1) & attacks);
                tables.lut2[pt][(int(from) << 6) | int(to)] =
                  static_cast<i16>(idx | (u16(from < to) << 15));
            }

            if (pt != PAWN)
                offset += constexpr_popcount(Attacks::PseudoAttacks[pt][from]);
            else if (from >= SQ_A2 && from <= SQ_H7)
                offset += constexpr_popcount(Attacks::PseudoAttacks[WHITE][from]);
        }

        offsets[pt] = offset;
    }

    IndexType cumulativeOffset = 0;
    for (Piece attacker : AllPieces)
    {
        const PieceType attackerType          = type_of(attacker);
        const IndexType cumulativePieceOffset = offsets[attackerType];
        const IndexType validTargets          = numValidTargets[attacker];

        for (Piece attacked : AllPieces)
        {
            bool      enemy        = (attacker ^ attacked) == 8;
            PieceType attackedType = type_of(attacked);

            int  map           = FullThreats::map[attackerType - 1][attackedType - 1];
            bool semi_excluded = attackerType == attackedType && (enemy || attackerType != PAWN);
            IndexType feature =
              cumulativeOffset
              + (color_of(attacked) * (validTargets / 2) + map) * cumulativePieceOffset;

            bool      excluded = map < 0;
            const u16 lo       = excluded ? FullThreats::Dimensions : feature;
            const u16 hi =
              (excluded || semi_excluded ? FullThreats::Dimensions : feature) - 0x8000u;
            tables.lut1[attacked][attacker] = u32(lo) | (u32(hi) << 16);
        }

        cumulativeOffset += validTargets * cumulativePieceOffset;
    }

    return tables;
}

constexpr ThreatLUTs threat_luts = init_threat_luts();

sf_always_inline IndexType resolve_threat_index(u32 l1, i16 v2) {
    return u16((v2 < 0 ? l1 >> 16 : l1) + u32(v2));
}

// Index of a feature for a given king position and another piece on some square
sf_always_inline IndexType FullThreats::make_index(
  Color perspective, Piece attacker, Square from, Square to, Piece attacked, Square ksq) {
    const unsigned orient  = (OrientTBL[ksq] ^ (56 * perspective)) * 65;
    const unsigned sq_pair = ((unsigned(from) << 6) | unsigned(to)) ^ orient;

    const u8 swap              = 8 * perspective;
    unsigned attacker_oriented = attacker ^ swap;
    unsigned attacked_oriented = attacked ^ swap;

    return resolve_threat_index(threat_luts.lut1[attacked_oriented][attacker_oriented],
                                threat_luts.lut2[type_of(attacker)][sq_pair]);
}

// Get a list of indices for active features in ascending order

void FullThreats::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square   ksq                = pos.square<KING>(perspective);
    const Bitboard occupied           = pos.pieces();
    const Bitboard pawnTargets        = pos.pieces(KNIGHT, ROOK);
    const Bitboard minorSliderTargets = pos.pieces(PAWN, KNIGHT, BISHOP, ROOK);
    const Bitboard queenTargets       = pos.pieces(PAWN, KNIGHT, BISHOP, ROOK, QUEEN);

    auto process_pawn_attacks = [&](Color c, Direction attkDir) {
        const Bitboard cPawns  = pos.pieces(c, PAWN);
        Bitboard       attacks = shift(cPawns, attkDir) & pawnTargets;
        while (attacks)
        {
            Square    to       = pop_lsb(attacks);
            Square    from     = to - attkDir;
            Piece     attacked = pos.piece_on(to);
            Piece     attacker = make_piece(c, PAWN);
            IndexType index    = make_index(perspective, attacker, from, to, attacked, ksq);
            active.push_back_if_lt(index, Dimensions);
        }
    };

    process_pawn_attacks(WHITE, NORTH_EAST);
    process_pawn_attacks(WHITE, NORTH_WEST);

    process_pawn_attacks(BLACK, SOUTH_WEST);
    process_pawn_attacks(BLACK, SOUTH_EAST);

    for (Color c : {WHITE, BLACK})
    {
        for (PieceType pt = KNIGHT; pt < KING; ++pt)
        {
            Piece    attacker = make_piece(c, pt);
            Bitboard bb       = pos.pieces(c, pt);
            Bitboard targets  = pt == KNIGHT || pt == QUEEN ? queenTargets : minorSliderTargets;
            while (bb)
            {
                Square   from    = pop_lsb(bb);
                Bitboard attacks = Attacks::attacks_bb(pt, from, occupied) & targets;
                while (attacks)
                {
                    Square    to       = pop_lsb(attacks);
                    Piece     attacked = pos.piece_on(to);
                    IndexType index    = make_index(perspective, attacker, from, to, attacked, ksq);
                    active.push_back_if_lt(index, Dimensions);
                }
            }
        }
    }
}

// Get a list of indices for recently changed features

void FullThreats::append_changed_indices(Color                   perspective,
                                         Square                  ksq,
                                         const DiffType&         diff,
                                         IndexList&              removed,
                                         IndexList&              added,
                                         const ThreatWeightType* prefetchBase,
                                         IndexType               prefetchStride) {
    const unsigned orient    = (OrientTBL[ksq] ^ (56 * perspective)) * 65;
    const unsigned pair_swap = 0x88 * perspective;
    const u32*     lut1_flat = &threat_luts.lut1[0][0];
    const i16*     lut2_flat = &threat_luts.lut2[0][0];

    for (const auto& dirty : diff.list)
    {
        const u32       raw   = dirty.raw();
        const unsigned  idx2  = (raw & 0x7FFFu) ^ orient;
        const unsigned  pair  = u8(raw >> 12) ^ pair_swap;
        const IndexType index = resolve_threat_index(lut1_flat[pair], lut2_flat[idx2]);

        if (prefetchBase)
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(index) * prefetchStride));

        if (raw >> 31)
            added.push_back_if_lt(index, Dimensions);
        else
            removed.push_back_if_lt(index, Dimensions);
    }
}

void FullThreats::append_changed_indices_both(Square                  white_ksq,
                                              Square                  black_ksq,
                                              const DiffType&         diff,
                                              IndexList&              white_removed,
                                              IndexList&              white_added,
                                              IndexList&              black_removed,
                                              IndexList&              black_added,
                                              const ThreatWeightType* prefetchBase,
                                              IndexType               prefetchStride) {
    const unsigned w_orient  = OrientTBL[white_ksq] * 65;
    const unsigned b_orient  = (OrientTBL[black_ksq] ^ 56) * 65;
    const u32*     lut1_flat = &threat_luts.lut1[0][0];
    const i16*     lut2_flat = &threat_luts.lut2[0][0];

    for (const auto& dirty : diff.list)
    {
        const u32      raw  = dirty.raw();
        const unsigned idx2 = raw & 0x7FFFu;
        const unsigned pair = u8(raw >> 12);

        const IndexType white_index =
          resolve_threat_index(lut1_flat[pair], lut2_flat[idx2 ^ w_orient]);
        const IndexType black_index =
          resolve_threat_index(lut1_flat[pair ^ 0x88], lut2_flat[idx2 ^ b_orient]);

        if (prefetchBase)
        {
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(white_index) * prefetchStride));
            prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(reinterpret_cast<const void*>(
              reinterpret_cast<uintptr_t>(prefetchBase) + uintptr_t(black_index) * prefetchStride));
        }

        if (raw >> 31)
        {
            white_added.push_back_if_lt(white_index, Dimensions);
            black_added.push_back_if_lt(black_index, Dimensions);
        }
        else
        {
            white_removed.push_back_if_lt(white_index, Dimensions);
            black_removed.push_back_if_lt(black_index, Dimensions);
        }
    }
}

}  // namespace Stockfish::Eval::NNUE::Features
