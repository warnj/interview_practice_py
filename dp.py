from typing import List

# https://leetcode.com/problems/minimum-cost-for-tickets
def mincostTicketsBrute(self, days: List[int], costs: List[int]) -> int:
    def minTickets(days, dayIndex, paidTo, costs, curCost):
        # paidTo is the next day we need to pay for a new ticket
        # dayIndex is current position in days
        while dayIndex < len(days) and days[dayIndex] < paidTo:
            dayIndex += 1  # find the next day we need to pay for
        if dayIndex >= len(days):  # have paid for all travel days
            return curCost
        return min(
            minTickets(days, dayIndex + 1, days[dayIndex] + 1, costs, curCost + costs[0]),
            minTickets(days, dayIndex + 1, days[dayIndex] + 7, costs, curCost + costs[1]),
            minTickets(days, dayIndex + 1, days[dayIndex] + 30, costs, curCost + costs[2])
        )
    return minTickets(days, 0, -1, costs, 0)
def mincostTickets(self, days: List[int], costs: List[int]) -> int:
    memo = {}  # store results of subproblems
    def minTickets(dayIndex, paidTo):
        if dayIndex >= len(days):
            return 0
        # Check if this state is already computed
        if (dayIndex, paidTo) in memo:
            return memo[(dayIndex, paidTo)]
        # If the current day is already paid for
        if days[dayIndex] < paidTo:
            result = minTickets(dayIndex + 1, paidTo)
        else:
            result = min(
                costs[0] + minTickets(dayIndex + 1, days[dayIndex] + 1),
                costs[1] + minTickets(dayIndex + 1, days[dayIndex] + 7),
                costs[2] + minTickets(dayIndex + 1, days[dayIndex] + 30)
            )
        memo[(dayIndex, paidTo)] = result
        return result
    return minTickets(0, 0)
def mincostTicketsEditorial(days, costs):
    lastDay = days[-1]
    dp = [0] * (lastDay + 1)  # store min cost to travel on all travel days up to dp[day]
    i = 0
    for day in range(1, lastDay + 1):
        # If we don't need to travel on this day, the cost won't change.
        if day < days[i]:
            dp[day] = dp[day - 1]
        else:
            # Buy a pass on this day, and move on to the next travel day.
            i += 1
            dp[day] = min(
                dp[day - 1] + costs[0],  # 1-day ticket
                dp[max(0, day - 7)] + costs[1],  # 7-day ticket
                dp[max(0, day - 30)] + costs[2]  # 30-day ticket
            )
    return dp[lastDay]
