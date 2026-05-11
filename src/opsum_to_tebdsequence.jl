struct OpSeqTerm
    bond::Pair{Int,Int}
    operator::ITensor
end

Base.iseven(ost::OpSeqTerm) = iseven(first(ost.bond))
Base.isodd(ost::OpSeqTerm) = isodd(first(ost.bond))
bond(ost::OpSeqTerm) = ost.bond
operator(ost::OpSeqTerm) = ost.operator

function isonesite(t::ITensors.Scaled{C,ITensors.Prod{ITensors.Op}}) where {C}
    return length(ITensors.sites(t)) == 1
end

function istwosite(t::ITensors.Scaled{C,ITensors.Prod{ITensors.Op}}) where {C}
    t_sites = sort(ITensors.sites(t))
    return length(t_sites) == 2 && first(t_sites) + 1 == last(t_sites)
end

function isonbond(t::ITensors.Scaled{C,ITensors.Prod{ITensors.Op}}, b::Pair{Int}) where {C}
    # `t` is a term in an OpSum
    # `t[n]` iterates over the factors (not including the scalar multiplier) of the operator
    # `ITensors.sites` returns the sites that the term acts on (a list of Ints)
    # For example, with `t = 1.0 Adag(1,) A(2,)` we have:
    # - `t[1]` = `Adag(1,)` and `t[2]` = `A(2,)`
    # - `ITensors.sites(t[1])` = 1 etc.
    t_sites = sort(ITensors.sites(t))
    @assert first(b) + 1 == last(b)
    errmsg =
        "the tebdsequence method only accepts 1-site operators or 2-site " *
        "operators on consecutive sites"
    if length(t_sites) == 1
        return only(t_sites) == first(b) || only(t_sites) == last(b)
    elseif length(t_sites) == 2
        if first(t_sites) + 1 == last(t_sites)
            return first(t_sites) == first(b) && last(t_sites) == last(b)
        else
            throw(ArgumentError(errmsg))
        end
    else
        throw(ArgumentError(errmsg))
    end
end

function tebdsequence(os::OpSum{T}, site_indices) where {T}
    N = length(site_indices)

    opseq = OpSeqTerm[]

    # 1. Split up the OpSum terms into (i, i+1) pairs.
    #    Remember to assign 1-site terms half on (i-1, i) and half on (i, i+1).
    #
    # 2. Create an ITensor operator for each term.

    for n in 1:(N - 1)
        terms_on_this_bond = filter(
            term -> isonbond(term, n => n+1), ITensors.LazyApply.terms(os)
        )
        # Skip this bond if there are no terms associated to it in the OpSum.
        if isempty(terms_on_this_bond)
            continue
        end

        ops_on_this_bond = []
        for x in terms_on_this_bond
            if isonesite(x)
                opn = only(ITensors.sites(x))
                Op = if 1 < opn < N
                    # Divide the term by two, because it will appear in another term of
                    # the sequence (the n-1 => n term if opn == n, or the n+1 => n+2 term if
                    # opn == n+1).
                    0.5 * ITensor(x, site_indices)
                else
                    # Single-site operators based on the edge of the MPS appear only once in
                    # the sequence.
                    ITensor(x, site_indices)
                end

                # Later we will need to sum all the operators in `ops_on_this_bond`
                # together. They will all need to be defined on the same site indices, i.e.
                # on both site_indices[n] and site_indices[n+1].  For this reason,
                # single-site operators must be properly expanded, with the identity on the
                # missing site.
                Op *= if opn == n
                    op(I, site_indices[n + 1])
                elseif opn == n+1
                    op(I, site_indices[n])
                end
                push!(ops_on_this_bond, Op)
            elseif istwosite(x)
                Op = ITensor(x, site_indices)
                push!(ops_on_this_bond, Op)
            else
                errmsg = "the tebdsequence method only accepts 1-site operators or 2-site operators"
                error(errmsg)
            end
        end

        push!(opseq, OpSeqTerm(n => n+1, sum(ops_on_this_bond)))
    end

    return opseq
end

function trotter1(os::OpSum{T}, sites, dt) where {T}
    opseq = tebdsequence(os, sites)
    oddseq = exp.(dt .* operator.(filter(isodd, opseq)))
    evenseq = exp.(dt .* operator.(filter(iseven, opseq)))
    return [oddseq, evenseq]
end

function extend_op(o::OpSeqTerm, sites)
    # Return an ITensor given by extending `o` with the identity on all sites on which it
    # wasn't already defined.
    op_sites = [first(bond(o)), last(bond(o))]
    missing_sites = setdiff(sites, sites[op_sites])
    return operator(o) * op(I, missing_sites)
end

function full_tensor(seq::Vector{OpSeqTerm}, sites)
    # Return the tensor given by adding together all the operators in `seq`.  This is mostly
    # for testing purposes.  Use with caution, and with a very small number of sites: it can
    # generate very big tensors.
    seq_extended = [extend_op(t, sites) for t in seq]

    # Extend all tensors so that they are defined on the whole list `sites`, then add them
    # all together.
    return sum(seq_extended)
end
